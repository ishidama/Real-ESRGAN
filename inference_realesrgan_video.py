# Real-ESRGANビデオ推論スクリプト
# ビデオ、画像、フォルダに対してReal-ESRGANによる超解像処理を実行します
import argparse
import cv2
import glob
import mimetypes
import numpy as np
import os
import shutil
import subprocess
import time
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from os import path as osp
from tqdm import tqdm

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# TODO アスペクト比がおかしい
# TODO コーデック選択ができるようにする
# TODO フレームレートの是正

# ffmpeg-pythonの動的インストール
try:
    import ffmpeg
except ImportError:
    import pip

    pip.main(["install", "--user", "ffmpeg-python"])
    import ffmpeg


def get_video_meta_info(video_path):
    """動画ファイルからメタ情報を取得する関数

    Args:
        video_path (str): 動画ファイルのパス

    Returns:
        dict: 動画のメタ情報を含む辞書（width, height, fps, audio, nb_frames, is_interlaced, field_order,
              display_aspect_ratio, sample_aspect_ratio, codec_name）

    Raises:
        RuntimeError: 動画解析に失敗した場合
    """
    try:
        ret = {}
        probe = ffmpeg.probe(video_path)

        # ビデオストリーム情報の取得
        video_streams = [
            stream for stream in probe["streams"] if stream["codec_type"] == "video"
        ]

        if not video_streams:
            raise RuntimeError(f"No video stream found in {video_path}")

        # オーディオの有無を確認
        has_audio = any(stream["codec_type"] == "audio" for stream in probe["streams"])

        # 動画コーデック名を取得
        if "codec_name" in video_streams[0]:
            ret["codec_name"] = video_streams[0]["codec_name"]
        else:
            ret["codec_name"] = "unknown"

        # アスペクト比情報の取得
        # 表示アスペクト比（DAR: Display Aspect Ratio）
        if "display_aspect_ratio" in video_streams[0]:
            ret["display_aspect_ratio"] = video_streams[0]["display_aspect_ratio"]
        elif "dar" in video_streams[0]:
            ret["display_aspect_ratio"] = video_streams[0]["dar"]
        elif "tags" in video_streams[0] and "DAR" in video_streams[0]["tags"]:
            ret["display_aspect_ratio"] = video_streams[0]["tags"]["DAR"]
        else:
            # 明示的なDARが指定されていない場合は、幅と高さから計算
            width = int(video_streams[0]["width"])
            height = int(video_streams[0]["height"])
            # gcdを使用して最大公約数を求め、アスペクト比を簡約
            import math

            gcd = math.gcd(width, height)
            ret["display_aspect_ratio"] = f"{width // gcd}:{height // gcd}"
            print(f"アスペクト比を計算: {ret['display_aspect_ratio']}")

        # サンプルアスペクト比（SAR: Sample Aspect Ratio - ピクセルのアスペクト比）
        if "sample_aspect_ratio" in video_streams[0]:
            ret["sample_aspect_ratio"] = video_streams[0]["sample_aspect_ratio"]
        else:
            ret["sample_aspect_ratio"] = "1:1"  # デフォルト：正方形ピクセル

        # 結果をセット
        ret["width"] = int(video_streams[0]["width"])
        ret["height"] = int(video_streams[0]["height"])

        # フレームレートの正確な処理
        if "avg_frame_rate" in video_streams[0]:
            fps_str = video_streams[0]["avg_frame_rate"]
            try:
                # 分数形式（例: "30000/1001"）の場合
                if "/" in fps_str:
                    num, den = map(int, fps_str.split("/"))
                    ret["fps"] = num / den
                    print(f"フレームレート: {fps_str} = {ret['fps']:.6f}fps")
                else:
                    # 小数形式の場合
                    ret["fps"] = float(fps_str)
                    print(f"フレームレート: {ret['fps']}fps")
            except Exception as e:
                print(f"フレームレート解析エラー: {e}, デフォルト値30を使用")
                ret["fps"] = 30.0
        else:
            ret["fps"] = 30.0  # デフォルト値
        ret["audio"] = ffmpeg.input(video_path).audio if has_audio else None
        # ret["nb_frames"] = int(video_streams[0]["nb_frames"])
        # nb_frames が 0 の場合は推定値を算出
        if "nb_frames" in video_streams[0]:
            nb_frames = int(video_streams[0]["nb_frames"])
            if nb_frames <= 0:  # 0または負の値の場合
                duration = (
                    float(probe["format"]["duration"])
                    if "format" in probe and "duration" in probe["format"]
                    else 0
                )
                fps = (
                    eval(video_streams[0]["avg_frame_rate"])
                    if "avg_frame_rate" in video_streams[0]
                    else 30
                )
                nb_frames = max(1, int(duration * fps))  # 最小値として1を保証
        else:
            # durationからフレーム数を計算
            duration = (
                float(probe["format"]["duration"])
                if "format" in probe and "duration" in probe["format"]
                else 0
            )
            fps = (
                eval(video_streams[0]["avg_frame_rate"])
                if "avg_frame_rate" in video_streams[0]
                else 30
            )
            nb_frames = max(1, int(duration * fps))  # 最小値として1を保証

        ret["nb_frames"] = nb_frames
        print(f"推定フレーム数: {nb_frames}")
        print(
            f"コーデック: {ret['codec_name']}, アスペクト比: {ret['display_aspect_ratio']}"
        )

        return ret
    except Exception as e:
        raise RuntimeError(f"Failed to probe video file {video_path}: {str(e)}")


class Reader:
    """
    入力データ（ビデオ、画像、フォルダ）を読み取るクラス
    ビデオストリームまたは画像ファイルからフレームを順次読み取る
    """

    def __init__(self, args):
        """
        Readerクラスの初期化

        Args:
            args: コマンドライン引数
        """
        self.args = args
        input_type = mimetypes.guess_type(args.input)[0]
        self.input_type = "folder" if input_type is None else input_type
        self.paths = []  # 画像・フォルダタイプ用のパスリスト
        self.audio = None
        self.input_fps = None

        # ビデオファイルの場合の処理
        if self.input_type.startswith("video"):
            video_path = args.input
            print(f"動画処理: {video_path}")

            # 動画のメタ情報を取得
            meta = get_video_meta_info(video_path)
            self.width = meta["width"]
            self.height = meta["height"]
            self.input_fps = meta["fps"]
            print(f"[fps debug] Reader: meta['fps'] = {self.input_fps}")
            self.audio = meta["audio"]
            self.nb_frames = meta["nb_frames"]

            # インターレース情報の処理
            input_stream = ffmpeg.input(video_path)

            # デインターレース処理の適用
            # 自動検出（auto）の場合は実データに基づいて決定
            apply_deinterlace = False
            deinterlace_filter = args.deinterlace

            if deinterlace_filter == "auto":
                # NOTE 自動判定モードはまだ未実装なので例外を投げて終了させる
                raise Exception("自動判定モードは未実装です")

                # if meta.get("is_interlaced", False):
                #     # インターレースが検出された場合、デフォルトのフィルターを適用
                #     deinterlace_filter = "bwdif"  # デフォルトでbwdifを使用
                #     apply_deinterlace = True
                #     print(
                #         f"インターレース映像を自動検出: デインターレースフィルター '{deinterlace_filter}' を適用"
                #     )
                # else:
                #     deinterlace_filter = (
                #         ""  # プログレッシブ映像なのでフィルタリング不要
                #     )
                #     print(
                #         "プログレッシブ映像を検出: デインターレース処理はスキップします"
                #     )
            elif deinterlace_filter:  # 明示的に指定された場合
                if (
                    not meta.get("is_interlaced", False)
                    and meta.get("field_order", "") == "progressive"
                ):
                    # 明示的にプログレッシブと判定された場合は警告を表示
                    print(
                        f"警告: プログレッシブ映像にデインターレースフィルター '{deinterlace_filter}' が指定されました"
                    )
                    print(
                        "  プログレッシブ映像にデインターレース処理を適用すると画質が低下する可能性があります"
                    )
                    print(
                        "  続行する場合は指定されたフィルターを適用します。自動判定を使用するには --deinterlace auto を指定してください"
                    )
                    # フィルターは適用する（ユーザーの明示的な指示を尊重）
                    apply_deinterlace = True
                else:
                    # インターレース映像または判定不能な場合は通常通り処理
                    apply_deinterlace = True
                    print(f"デインターレースフィルター '{deinterlace_filter}' を適用")

            # デインターレースフィルターの適用
            if apply_deinterlace and deinterlace_filter != "NONE":
                if deinterlace_filter == "yadif":
                    # yadif = Yet Another DeInterlacing Filter
                    input_stream = input_stream.filter("yadif", mode=0, parity=-1)
                    print(f"デインターレース処理を適用しました: {deinterlace_filter}")
                elif deinterlace_filter == "bwdif":
                    # bwdif = Bob Weaver DeInterlacing Filter
                    input_stream = input_stream.filter("bwdif", mode=0, parity=-1)
                    print(f"デインターレース処理を適用しました: {deinterlace_filter}")
                elif deinterlace_filter == "w3fdif":
                    # w3fdif = 3-field deinterlacing filter
                    input_stream = input_stream.filter("w3fdif")
                    print(f"デインターレース処理を適用しました: {deinterlace_filter}")
            else:
                # 指定のためフィルタリング不要
                print(
                    f"指定のためデインターレース処理はスキップ : {deinterlace_filter}"
                )

            # SAR補正（ピクセルアスペクト比が1:1でない場合は正方ピクセル化）
            sar = meta.get("sample_aspect_ratio", "1:1")
            if sar != "1:1":
                try:
                    sar_num, sar_den = map(int, sar.split(":"))
                    new_width = int(round(self.width * sar_num / sar_den))
                    input_stream = input_stream.filter("scale", new_width, self.height)
                    input_stream = input_stream.filter(
                        "setsar", 1
                    )  # ffmpeg setsarは整数でOK
                    print(
                        f"SAR補正: {sar} → 1:1, リサイズ後: {new_width}x{self.height}"
                    )
                    self.width = new_width  # 後続処理のため幅も更新
                except Exception as e:
                    print(f"SAR補正処理中にエラー: {e}")

            # ffmpegを使用してパイプ経由でフレームを読み込む設定
            self.stream_reader = input_stream.output(
                "pipe:", format="rawvideo", pix_fmt="bgr24", loglevel="error"
            ).run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin)

            print(
                f"動画情報: {self.width}x{self.height}, {self.input_fps}fps, {self.nb_frames}フレーム"
            )

        else:
            # 画像またはフォルダの場合の処理
            if self.input_type.startswith("image"):
                self.paths = [args.input]
            else:
                self.paths = sorted(glob.glob(os.path.join(args.input, "*")))

            self.nb_frames = len(self.paths)
            assert self.nb_frames > 0, "empty folder"
            from PIL import Image

            tmp_img = Image.open(self.paths[0])
            self.width, self.height = tmp_img.size
        self.idx = 0

    def get_resolution(self):
        """解像度（高さ、幅）を取得"""
        return self.height, self.width

    def get_fps(self):
        """出力フレームレートを決定して返す

        コマンドライン引数で指定されたfpsを優先、
        指定がなければ元の動画のfps、それも取得できない場合はデフォルト値を使用。

        Returns:
            float: フレームレート
        """
        if self.args.fps is not None:
            print(
                f"[fps debug] Reader.get_fps() returns user-specified {self.args.fps}"
            )
            return self.args.fps
        elif self.input_fps is not None:
            print(f"[fps debug] Reader.get_fps() returns original {self.input_fps}")
            return self.input_fps
        print("[fps debug] Reader.get_fps() fallback to 24")
        return 24  # デフォルト値

    def get_audio(self):
        """音声ストリームを取得"""
        return self.audio

    def __len__(self):
        """総フレーム数を取得"""
        return self.nb_frames

    def get_frame_from_stream(self):
        """ビデオストリームからフレームを取得"""
        img_bytes = self.stream_reader.stdout.read(
            self.width * self.height * 3
        )  # 1ピクセルあたり3バイト
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        return img

    def get_frame_from_list(self):
        """ファイルリストからフレームを取得"""
        if self.idx >= self.nb_frames:
            return None
        img = cv2.imread(self.paths[self.idx])
        self.idx += 1
        return img

    def get_frame(self):
        """入力タイプに応じてフレームを取得"""
        if self.input_type.startswith("video"):
            return self.get_frame_from_stream()
        else:
            return self.get_frame_from_list()

    def close(self):
        """リソースを解放"""
        if self.input_type.startswith("video"):
            self.stream_reader.stdin.close()
            self.stream_reader.wait()


class Writer:
    """処理したフレームを動画ファイルに書き込むクラス

    ffmpegを使用して、処理されたフレームをパイプ経由で動画ファイルに書き込みます。
    音声トラックがある場合はそれも保持されます。
    ソース動画のアスペクト比も維持されます。
    """

    def __init__(self, args, audio, height, width, video_save_path, fps):
        """ライターの初期化

        Args:
            args: コマンドライン引数
            audio: 音声ストリーム（または None）
            height (int): 入力フレームの高さ
            width (int): 入力フレームの幅
            video_save_path (str): 出力動画のファイルパス
            fps (float): フレームレート
        """
        # 出力解像度の計算
        out_width = int(width * args.outscale)
        out_height = int(height * args.outscale)

        # 4K以上の解像度に関する警告
        if out_height > 2160:
            print(
                "警告: 4K以上の解像度の動画を生成しています。IO速度によって非常に遅くなる可能性があります。",
                f"現在の出力解像度: {out_width}x{out_height}",
                "outscale パラメータ(-s)を小さくすることを推奨します。",
            )

        try:
            # 出力設定の基本部分（共通）
            input_stream = ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="bgr24",
                s=f"{out_width}x{out_height}",
                framerate=fps,
            )

            # 入力動画からアスペクト比情報を取得（存在する場合）
            display_aspect_ratio = None
            if hasattr(args, "input") and os.path.exists(args.input):
                try:
                    meta = get_video_meta_info(args.input)
                    if "display_aspect_ratio" in meta:
                        display_aspect_ratio = meta["display_aspect_ratio"]
                        print(f"元の表示アスペクト比を取得: {display_aspect_ratio}")
                except Exception as e:
                    print(f"アスペクト比情報の取得中にエラーが発生しました: {e}")

            # アスペクト比を設定（存在する場合）
            if display_aspect_ratio:
                # ffmpegのフィルタとしてアスペクト比を設定
                input_stream = input_stream.filter(
                    "setdar", display_aspect_ratio.replace(":", "/")
                )
                print(f"元の表示アスペクト比を維持: {display_aspect_ratio}")

            # ビデオコーデック設定
            video_codec_args = {
                "pix_fmt": "yuv420p",
                "preset": "medium",  # エンコード速度と品質のバランス
                "loglevel": "error",
            }

            # コーデック選択（H.264またはHEVC/H.265）
            if hasattr(args, "codec") and args.codec == "hevc":
                video_codec_args["vcodec"] = "libx265"
                video_codec_args["crf"] = "18"  # H.265では23が標準的なH.264の23相当品質
                print(
                    "HEVC/H.265コーデックを使用します（高圧縮・高品質、エンコードに時間がかかります）"
                )
            else:  # h264
                video_codec_args["vcodec"] = "libx264"
                video_codec_args["crf"] = "23"  # H.264の標準品質

            # 音声がある場合とない場合で出力設定を変更
            if audio is not None:
                print(f"音声トラックを出力動画に含めます: {video_save_path}")
                self.stream_writer = (
                    input_stream.output(
                        audio,
                        video_save_path,
                        **video_codec_args,
                        acodec="copy",  # 音声はそのままコピー
                        r=fps,  # フレームレートを指定
                    )
                    .overwrite_output()
                    .run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin)
                )
            else:
                print(f"音声なしで出力動画を生成します: {video_save_path}")
                self.stream_writer = (
                    input_stream.output(
                        video_save_path,
                        **video_codec_args,
                    )
                    .overwrite_output()
                    .run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin)
                )

        except Exception as e:
            raise RuntimeError(f"動画ライターの初期化に失敗しました: {e}")

    def write_frame(self, frame):
        """1フレームを動画ファイルに書き込む

        Args:
            frame (np.ndarray): 書き込むフレーム画像
        """
        try:
            # numpyの配列をバイト列に変換
            frame_bytes = frame.astype(np.uint8).tobytes()
            self.stream_writer.stdin.write(frame_bytes)
        except Exception as e:
            print(f"フレーム書き込み中にエラーが発生しました: {e}")

    def close(self):
        """ライターを閉じてリソースを解放する"""
        try:
            self.stream_writer.stdin.close()
            self.stream_writer.wait()
            print("動画の書き込みが完了しました")
        except Exception as e:
            print(f"ライターのクローズ中にエラーが発生しました: {e}")


def inference_video(args, video_save_path, device=None):
    """
    動画にリアルESRGANを適用して超解像処理を行う関数。

    引数:
        args: コマンドライン引数を含むオブジェクト。
            model_name: 使用するモデルの名前
            tile: タイル処理のサイズ
            tile_pad: タイルのパディングサイズ
            pre_pad: 前処理のパディングサイズ
            fp32: 32ビット浮動小数点精度を使用するかのフラグ
            face_enhance: 顔の強調処理を適用するかのフラグ
            outscale: 出力スケール
            denoise_strength: ノイズ除去の強さ (0-1)
        video_save_path: 処理後の動画を保存するパス
        device: 計算に使用するデバイス (CPU/GPU)。None の場合は自動選択
        total_workers: 並列処理時の全ワーカー数
        worker_idx: 現在のワーカーのインデックス

    処理の流れ:
        1. 指定されたモデル名に基づいて適切なモデルをロード
        2. モデルが存在しない場合はGitHubからダウンロード
        3. RealESRGANのアップサンプラーを初期化
        4. 顔の強調処理が指定されている場合はGFPGANを初期化
        5. 入力動画のフレームを順次読み込み、超解像処理を適用
        6. 処理済みのフレームを出力動画に書き込み
        7. 処理の進捗状況をリアルタイムで表示（FPS、残り時間など）

    注意:
        - アニメモデルでは顔の強調処理は無効
        - CUDAメモリ不足の場合はtileパラメータを調整する必要あり
        - 処理速度はハードウェアとモデルに依存
    """
    # ---------------------- determine models according to model names ---------------------- #
    args.model_name = args.model_name.split(".pth")[0]
    if args.model_name == "RealESRGAN_x4plus":  # x4 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        ]
    elif args.model_name == "RealESRNet_x4plus":  # x4 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
        ]
    elif (
        args.model_name == "RealESRGAN_x4plus_anime_6B"
    ):  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        ]
    elif args.model_name == "RealESRGAN_x2plus":  # x2 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        netscale = 2
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        ]
    elif args.model_name == "realesr-animevideov3":  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=16,
            upscale=4,
            act_type="prelu",
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
        ]
    elif args.model_name == "realesr-general-x4v3":  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        ]

    # ---------------------- determine model paths ---------------------- #
    model_path = os.path.join("weights", args.model_name + ".pth")
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url,
                model_dir=os.path.join(ROOT_DIR, "weights"),
                progress=True,
                file_name=None,
            )

    # use dni to control the denoise strength
    dni_weight = None
    if args.model_name == "realesr-general-x4v3" and args.denoise_strength != 1:
        # denoise strengthが1ではない場合、WDNモデルを使用する
        wdn_model_path = model_path.replace(
            "realesr-general-x4v3", "realesr-general-wdn-x4v3"
        )
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        device=device,
    )

    if "anime" in args.model_name and args.face_enhance:
        print(
            "face_enhance is not supported in anime models, we turned this option off for you. "
            "if you insist on turning it on, please manually comment the relevant lines of code."
            "\n注意: アニメモデルでは顔強調機能はサポートされていないため、無効化しました。"
            "強制的に有効にしたい場合は、該当コードを手動で変更してください。"
        )
        args.face_enhance = False

    if args.face_enhance:  # Use GFPGAN for face enhancement
        try:
            print("顔強調機能を有効化します（GFPGANモデルをロード中...）")
            from gfpgan import GFPGANer

            face_enhancer = GFPGANer(
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                upscale=args.outscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=upsampler,
            )  # TODO support custom device
            print("GFPGAN顔強調モデルのロードが完了しました")
        except Exception as e:
            print("GFPGAN顔強調モデルのロード中にエラーが発生しました: " + str(e))
            print("顔強調機能を無効化します")
            face_enhancer = None
            args.face_enhance = False
    else:
        face_enhancer = None

    reader = Reader(args)
    audio = reader.get_audio()
    height, width = reader.get_resolution()
    fps = reader.get_fps()
    writer = Writer(args, audio, height, width, video_save_path, fps)

    # 進捗状況追跡のための変数
    total_frames = max(1, len(reader))  # 最小値として1を保証
    processed_frames = 0
    processing_times = []  # フレームごとの処理時間を記録

    print(
        "\n動画処理を開始します - モデル: "
        + args.model_name
        + ", スケール: "
        + str(args.outscale)
        + "倍, タイル: "
        + str(args.tile)
    )

    # プログレスバーの設定（フォーマットをカスタマイズ）
    pbar = tqdm(
        total=total_frames,
        unit="frame",
        desc="処理中",
        bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} フレーム "
        + "[経過: {elapsed}, 残り: {remaining}, {rate_fmt}{postfix}]",
    )

    while True:
        frame_start = time.time()
        img = reader.get_frame()
        if img is None:
            break

        try:
            if args.face_enhance and face_enhancer is not None:
                _, _, output = face_enhancer.enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True
                )
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print("エラー:", error)
            print("CUDA メモリ不足の場合は、--tile パラメータを小さくしてください。")
        else:
            writer.write_frame(output)

        # デバイスに応じた同期処理
        if device is not None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "mps":
                torch.mps.synchronize()  # Apple Silicon用の同期

        # フレーム処理時間の計算と記録
        frame_time = time.time() - frame_start
        processing_times.append(frame_time)
        processed_frames += 1

        # 過去20フレームの平均処理時間から現在のFPSを計算
        recent_times = processing_times[-min(20, len(processing_times)) :]
        current_fps = (
            1.0 / (sum(recent_times) / len(recent_times)) if recent_times else 0
        )

        # 残りフレーム数と現在の処理速度から残り時間を推定
        frames_left = total_frames - processed_frames
        estimated_time_left = frames_left / current_fps if current_fps > 0 else 0
        hours, remainder = divmod(estimated_time_left, 3600)
        minutes, seconds = divmod(remainder, 60)

        # 進捗情報の更新
        progress_percent = (
            (processed_frames / total_frames) * 100 if total_frames > 0 else 0
        )
        est_time_str = (
            f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            if current_fps > 0
            else "計算中..."
        )

        pbar.set_postfix(
            {
                "FPS": f"{current_fps:.2f}",
                "進捗": f"{progress_percent:.1f}%",
                "推定残時間": est_time_str,
            }
        )
        pbar.update(1)

    reader.close()
    writer.close()


def run(args):
    """Mac向けシンプル処理関数 - 並列処理なし、MPS最適化"""
    args.video_name = osp.splitext(os.path.basename(args.input))[0]
    video_save_path = osp.join(args.output, f"{args.video_name}_{args.suffix}.mp4")

    if args.extract_frame_first:
        tmp_frames_folder = osp.join(args.output, f"{args.video_name}_inp_tmp_frames")
        os.makedirs(tmp_frames_folder, exist_ok=True)
        subprocess.run(
            [
                args.ffmpeg_bin,
                "-i",
                args.input,
                "-qscale:v",
                "1",
                "-qmin",
                "1",
                "-qmax",
                "1",
                "-vsync",
                "0",
                f"{tmp_frames_folder}/frame%08d.png",
            ],
            check=True,
        )
        args.input = tmp_frames_folder

    # Mac向けシンプルなデバイス設定
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple Silicon GPU (M1/M2/M3) を使用して処理を実行します")
        if args.fp32:
            print(
                "注意: MPS でfp32を使用します。精度は向上しますが処理速度が低下する場合があります"
            )
        else:
            print(
                "注意: MPS でfp16を使用します。処理が不安定な場合は --fp32 を追加してください"
            )
    else:
        device = torch.device("cpu")
        print("CPU を使用して処理を実行します（処理に時間がかかります）")

    # 単一プロセスで処理を実行
    print("単一プロセスで処理を開始します（Mac最適化）")
    inference_video(args, video_save_path, device=device)

    # MKV入力時の全トラック保持再パッケージ処理
    if args.input.lower().endswith(".mkv"):
        final_mkv_path = osp.join(args.output, f"{args.video_name}_{args.suffix}.mkv")
        print("mkv入力なので、動画以外の全トラックを保持してmkvで出力します")
        mkv_cmd = [
            args.ffmpeg_bin,
            "-i",
            args.input,
            "-i",
            video_save_path,
            "-map",
            "1:v:0",  # 処理済み動画
            "-map",
            "0:a?",  # 元の音声
            "-map",
            "0:s?",  # 元の字幕
            "-map",
            "0:t?",  # 元のチャプター等
            "-c",
            "copy",
            "-y",
            final_mkv_path,
        ]
        print("mkv再パッケージコマンド: " + " ".join(mkv_cmd))
        subprocess.run(mkv_cmd, check=True)
        print("最終出力: " + final_mkv_path)
        # mp4一時ファイルを削除
        os.remove(video_save_path)

    print("処理が完了しました")


def main():
    """
    Real-ESRGANの推論デモ
    主にアニメビデオの復元に使用される
    """
    parser = argparse.ArgumentParser()
    # 入力関連のオプション
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="inputs",
        help="入力ビデオ、画像、またはフォルダ",
    )
    parser.add_argument(
        "-n",
        "--model_name",
        type=str,
        default="realesr-general-x4v3",
        help=(
            "モデル名: realesr-animevideov3 | RealESRGAN_x4plus_anime_6B | RealESRGAN_x4plus | RealESRNet_x4plus |"
            " RealESRGAN_x2plus | realesr-general-x4v3"
            " デフォルト:realesr-animevideov3"
        ),
    )
    parser.add_argument(
        "-o", "--output", type=str, default="results", help="出力フォルダ"
    )

    # 処理関連のオプション
    parser.add_argument(
        "-dn",
        "--denoise_strength",
        type=float,
        default=0.5,
        help=(
            "デノイズ強度。0=弱いデノイズ（ノイズを保持）、1=強いデノイズ能力。"
            "realesr-general-x4v3モデルでのみ使用"
        ),
    )
    parser.add_argument(
        "-s",
        "--outscale",
        type=float,
        default=4,
        help="最終的な画像のアップサンプリング倍率",
    )
    parser.add_argument(
        "--suffix", type=str, default="out", help="復元されたビデオのサフィックス"
    )

    # タイル関連のオプション
    parser.add_argument(
        "-t",
        "--tile",
        type=int,
        default=0,
        help="タイルサイズ、テスト時にタイルを使用しない場合は0",
    )
    parser.add_argument("--tile_pad", type=int, default=10, help="タイルパディング")
    parser.add_argument(
        "--pre_pad", type=int, default=0, help="各境界での前パディングサイズ"
    )

    # 拡張機能のオプション
    parser.add_argument(
        "--face_enhance", action="store_true", help="GFPGANを使用して顔を強化"
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="推論時にfp32精度を使用。デフォルト: fp16（半精度）",
    )

    # ビデオ関連のオプション
    parser.add_argument("--fps", type=float, default=None, help="出力ビデオのFPS")
    parser.add_argument("--ffmpeg_bin", type=str, default="ffmpeg", help="ffmpegのパス")
    parser.add_argument(
        "--extract_frame_first", action="store_true", help="最初にフレームを抽出"
    )
    parser.add_argument(
        "--num_process_per_gpu", type=int, default=1, help="GPU当たりのプロセス数"
    )

    # 追加オプション
    parser.add_argument(
        "--alpha_upsampler",
        type=str,
        default="realesrgan",
        help="アルファチャンネル用のアップサンプラー。オプション: realesrgan | bicubic",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="auto",
        help="画像拡張子。オプション: auto | jpg | png、autoは入力と同じ拡張子を使用",
    )

    parser.add_argument(
        "--codec",
        type=str,
        default="hevc",
        choices=["avc", "hevc"],
        help="Video codec for output. h264 is more compatible, hevc (H.265) has better compression but slower encoding (出力動画のコーデック。h264は互換性が高く、hevc (H.265)は圧縮率が高いが処理が遅い)",
    )
    parser.add_argument(
        "--deinterlace",
        type=str,
        default="bwdif",
        choices=["NONE", "yadif", "bwdif", "w3fdif", "auto"],
        help="Deinterlace filter for interlaced videos. yadif=standard, bwdif=higher quality but slower, w3fdif=highest quality (インターレース映像の解除フィルター。yadif=標準、bwdif=高品質、w3fdif=最高品質)",
    )

    args = parser.parse_args()

    # 入力パスの正規化
    args.input = args.input.rstrip("/").rstrip("\\")
    os.makedirs(args.output, exist_ok=True)

    # 入力タイプの判定（ビデオかどうか）
    mime_type = mimetypes.guess_type(args.input)[0]
    if mime_type is not None and mime_type.startswith("video"):
        is_video = True
    else:
        is_video = False

    # ビデオでない場合はフレーム抽出オプションを無効化
    if args.extract_frame_first and not is_video:
        args.extract_frame_first = False

    # メイン処理の実行
    run(args)

    # フレーム抽出後の一時フォルダ削除
    if args.extract_frame_first:
        tmp_frames_folder = osp.join(args.output, f"{args.video_name}_inp_tmp_frames")
        shutil.rmtree(tmp_frames_folder)


if __name__ == "__main__":
    """
    スクリプトが直接実行された場合のエントリーポイント
    メイン関数を呼び出してReal-ESRGAN処理を開始する
    """
    main()
