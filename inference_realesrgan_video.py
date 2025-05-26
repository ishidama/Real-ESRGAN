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
        ret["fps"] = eval(
            video_streams[0]["avg_frame_rate"]
        )  # 例: "30000/1001" → 29.97
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


def get_sub_video(args, num_process, process_idx):
    """動画を分割して処理するための部分動画を作成する

    Args:
        args: コマンドライン引数
        num_process (int): 並列処理数
        process_idx (int): 現在の処理インデックス

    Returns:
        str: 作成された部分動画のファイルパス

    機能:
        - プログレッシブ映像の場合はストリームコピーを使用して再エンコードを回避
        - インターレース映像には高品質エンコード設定を適用
        - オーディオは常にコピーし再エンコードしない
        - 表示アスペクト比を維持
    """
    # 単一プロセスの場合は分割不要
    if num_process == 1:
        return args.input

    # 動画のメタ情報を取得
    try:
        meta = get_video_meta_info(args.input)
        duration = int(meta["nb_frames"] / meta["fps"])
        part_time = duration // num_process
        print(f"動画情報: 総時間 {duration}秒, 分割時間 {part_time}秒")

        # 出力ディレクトリの準備
        tmp_dir = osp.join(args.output, f"{args.video_name}_inp_tmp_videos")
        os.makedirs(tmp_dir, exist_ok=True)

        out_path = osp.join(tmp_dir, f"{process_idx:03d}.mp4")

        # ffmpegコマンドの構築
        start_time = part_time * process_idx
        end_time = (
            part_time * (process_idx + 1) if process_idx != num_process - 1 else ""
        )

        # 高精度なシーク用に先行入力オプションを追加
        cmd = [
            args.ffmpeg_bin,
            "-nostdin",  # 標準入力を無効化
            f"-i {args.input}",
            "-ss",
            f"{start_time}",
        ]

        if end_time:
            cmd.extend(["-to", f"{end_time}"])

        # インターレース情報の処理（引数で明示的に指定された場合のみ）
        deinterlace_filter = args.deinterlace
        need_deinterlace = False
        video_filters = []

        # デインターレースフィルタの処理（明示的な指定のみ）
        if deinterlace_filter:  # 明示的にフィルタが指定された場合のみ
            need_deinterlace = True
            if deinterlace_filter == "yadif":
                video_filters.append("yadif=mode=0:parity=-1")
                print("yadifデインターレースフィルタを適用")
            elif deinterlace_filter == "bwdif":
                video_filters.append("bwdif=mode=0:parity=-1")
                print("bwdifデインターレースフィルタを適用")
            elif deinterlace_filter == "w3fdif":
                video_filters.append("w3fdif")
                print("w3fdifデインターレースフィルタを適用")
            else:
                print(f"不明なデインターレースフィルタ: {deinterlace_filter}")
                need_deinterlace = False

        # 表示アスペクト比の設定
        if "display_aspect_ratio" in meta:
            # アスペクト比を設定 (setdarフィルタを使用)
            video_filters.append(
                f"setdar={meta['display_aspect_ratio'].replace(':', '/')}"
            )
            print(f"アスペクト比を設定: {meta['display_aspect_ratio']}")

        # フィルタの適用
        if video_filters:
            cmd.extend(["-vf", ",".join(video_filters)])

        # 映像コーデック設定
        if need_deinterlace:
            # インターレース映像の場合は高品質エンコード設定を適用
            cmd.extend(
                [
                    "-c:v",
                    "libx264",  # ビデオコーデックを明示的に指定
                    "-crf",
                    "0",  # 高品質設定 (0-51、数値が低いほど高品質)
                    "-preset",
                    "slower",  # エンコード品質優先設定
                    "-tune",
                    "film",  # フィルム向け調整
                ]
            )
            print("インターレース映像用の高品質エンコード設定を適用")
        else:
            # プログレッシブ映像の場合はストリームコピー
            cmd.extend(["-c:v", "copy"])
            print("プログレッシブ映像用のストリームコピーを適用（再エンコードなし）")

        # オーディオコーデックの設定（常にコピー）
        cmd.extend(["-c:a", "copy"])

        # 非同期モードの設定とファイル出力
        cmd.extend(
            [
                "-async",
                "1",
                out_path,
                "-y",
            ]
        )

        # コマンドの実行
        print(f"実行中: {' '.join(cmd)}")
        result = subprocess.run(" ".join(cmd), shell=True, capture_output=True)

        if result.returncode != 0:
            print(f"警告: 動画分割中にエラーが発生しました: {result.stderr.decode()}")
            # エラー出力の詳細をログ
            print(f"詳細: {result.stderr.decode()[:500]}...")

        return out_path
    except Exception as e:
        print(f"エラー: 動画分割中に例外が発生しました: {e}")
        return args.input  # エラー時は入力動画をそのまま返す


class Reader:
    """
    入力データ（ビデオ、画像、フォルダ）を読み取るクラス
    ビデオストリームまたは画像ファイルからフレームを順次読み取る
    """

    def __init__(self, args, total_workers=1, worker_idx=0):
        """
        Readerクラスの初期化

        Args:
            args: コマンドライン引数
            total_workers: 総ワーカー数
            worker_idx: 現在のワーカーインデックス
        """
        self.args = args
        input_type = mimetypes.guess_type(args.input)[0]
        self.input_type = "folder" if input_type is None else input_type
        self.paths = []  # 画像・フォルダタイプ用のパスリスト
        self.audio = None
        self.input_fps = None

        # ビデオファイルの場合の処理
        if self.input_type.startswith("video"):
            video_path = get_sub_video(
                args, total_workers, worker_idx
            )  # TODO PIPE経由にしたい
            self.stream_reader = (
                ffmpeg.input(video_path)
                .output("pipe:", format="rawvideo", pix_fmt="bgr24", loglevel="error")
                .run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin)
            )
            meta = get_video_meta_info(video_path)
            self.width = meta["width"]
            self.height = meta["height"]
            self.input_fps = meta["fps"]
            self.audio = meta["audio"]
            self.nb_frames = meta["nb_frames"]

        else:
            # 画像またはフォルダの場合の処理
            if self.input_type.startswith("image"):
                self.paths = [args.input]
            else:
                paths = sorted(glob.glob(os.path.join(args.input, "*")))
                tot_frames = len(paths)
                num_frame_per_worker = tot_frames // total_workers + (
                    1 if tot_frames % total_workers else 0
                )
                self.paths = paths[
                    num_frame_per_worker * worker_idx : num_frame_per_worker
                    * (worker_idx + 1)
                ]

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
        """FPS（フレームレート）を取得"""
        if self.args.fps is not None:
            return self.args.fps
        elif self.input_fps is not None:
            return self.input_fps
        return 24

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
    """
    処理済みフレームをビデオファイルに書き出すクラス
    ffmpegを使用してエンコードとファイル出力を行う
    """

    def __init__(self, args, audio, height, width, video_save_path, fps):
        """
        Writerクラスの初期化

        Args:
            args: コマンドライン引数
            audio: 音声ストリーム
            height: 入力画像の高さ
            width: 入力画像の幅
            video_save_path: 出力ビデオファイルのパス
            fps: フレームレート
        """
        # 出力解像度の計算（スケール倍率を適用）
        out_width, out_height = int(width * args.outscale), int(height * args.outscale)
        if out_height > 2160:
            print(
                "You are generating video that is larger than 4K, which will be very slow due to IO speed.",
                "We highly recommend to decrease the outscale(aka, -s).",
            )

        # 音声ありの場合のffmpeg設定
        if audio is not None:
            self.stream_writer = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="bgr24",
                    s=f"{out_width}x{out_height}",
                    framerate=fps,
                )
                .output(
                    audio,
                    video_save_path,
                    pix_fmt="yuv420p",
                    vcodec="libx264",
                    loglevel="error",
                    acodec="copy",
                )
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin)
            )
        else:
            # 音声なしの場合のffmpeg設定
            self.stream_writer = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="bgr24",
                    s=f"{out_width}x{out_height}",
                    framerate=fps,
                )
                .output(
                    video_save_path,
                    pix_fmt="yuv420p",
                    vcodec="libx264",
                    loglevel="error",
                )
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin)
            )

    def write_frame(self, frame):
        """フレームをビデオストリームに書き込み"""
        frame = frame.astype(np.uint8).tobytes()
        self.stream_writer.stdin.write(frame)

    def close(self):
        """ストリームを閉じてリソースを解放"""
        self.stream_writer.stdin.close()
        self.stream_writer.wait()


def inference_video(args, video_save_path, device=None, total_workers=1, worker_idx=0):
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

    reader = Reader(args, total_workers, worker_idx)
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
    """
    実行のメイン関数
    入力の前処理、マルチプロセス処理、後処理を行う

    Args:
        args: コマンドライン引数
    """
    args.video_name = osp.splitext(os.path.basename(args.input))[0]
    video_save_path = osp.join(args.output, f"{args.video_name}_{args.suffix}.mp4")

    # フレーム抽出が指定されている場合の前処理
    if args.extract_frame_first:
        tmp_frames_folder = osp.join(args.output, f"{args.video_name}_inp_tmp_frames")
        os.makedirs(tmp_frames_folder, exist_ok=True)
        os.system(
            f"ffmpeg -i {args.input} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  {tmp_frames_folder}/frame%08d.png"
        )
        args.input = tmp_frames_folder

    # デバイスの検出と設定
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device_type = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon (M1/M2/M3) MPSの場合は1つのGPUとしてカウント
        num_gpus = 1
        device_type = "mps"
        print("Using MPS device (Apple Silicon GPU)")
    else:
        num_gpus = 0
        device_type = "cpu"
        print("Using CPU for processing")

    # プロセス数の計算
    num_process = max(1, num_gpus * args.num_process_per_gpu)
    print(f"処理に使用するプロセス数: {num_process}")

    # シングルプロセスの場合
    if num_process == 1:
        # デバイスの選択
        if device_type == "cuda":
            device = torch.device("cuda:0")
            print("NVIDIA GPU (CUDA) を使用して処理を実行します")
        elif device_type == "mps":
            device = torch.device("mps")
            print("Apple Silicon GPU (M1/M2/M3) を使用して処理を実行します")
            if args.fp32:
                print(
                    "注意: MPS (Apple Silicon) でfp32を使用します。精度は向上しますが処理速度が低下する場合があります"
                )
            else:
                print(
                    "注意: MPS (Apple Silicon) でfp16を使用します。処理が不安定な場合は --fp32 を追加してください"
                )
        else:
            device = torch.device("cpu")
            print("CPU を使用して処理を実行します（処理に時間がかかります）")

        inference_video(args, video_save_path, device=device)
        # ↓↓↓ ここでmkv再パッケージ処理を必ず呼ぶ
        if args.input.lower().endswith(".mkv"):
            final_mkv_path = osp.join(
                args.output, f"{args.video_name}_{args.suffix}.mkv"
            )
            print("mkv入力なので、動画以外の全トラックを保持してmkvで出力します")
            mkv_cmd = [
                args.ffmpeg_bin,
                "-i",
                args.input,
                "-i",
                video_save_path,
                "-map",
                "1:v:0",
                "-map",
                "0:a?",
                "-map",
                "0:s?",
                "-map",
                "0:t?",
                "-c",
                "copy",
                "-y",
                final_mkv_path,
            ]
            print("mkv再パッケージコマンド: " + " ".join(mkv_cmd))
            subprocess.run(mkv_cmd, check=True)
            print("最終出力: " + final_mkv_path)
            # os.remove(video_save_path)
        return

    print(f"マルチプロセス処理を開始します（{num_process}プロセス）")
    ctx = torch.multiprocessing.get_context("spawn")
    pool = ctx.Pool(num_process)
    os.makedirs(
        osp.join(args.output, f"{args.video_name}_out_tmp_videos"), exist_ok=True
    )
    pbar = tqdm(total=num_process, unit="sub_video", desc="処理中")

    # マルチプロセス処理
    for i in range(num_process):
        sub_video_save_path = osp.join(
            args.output, f"{args.video_name}_out_tmp_videos", f"{i:03d}.mp4"
        )

        # デバイスの選択
        if device_type == "cuda":
            # CUDA: 複数GPUがある場合は分散
            device = torch.device(f"cuda:{i % num_gpus}")
        elif device_type == "mps" and i == 0:
            # MPS: Apple Siliconの場合は最初のプロセスのみGPUを使用（MPSは現在マルチプロセスでのGPU共有に制限あり）
            device = torch.device("mps")
            print(
                "警告: Apple Silicon GPUでは1プロセスのみGPUを使用し、残りはCPUで実行されます"
            )
        elif device_type == "mps":
            # MPS: 残りのプロセスはCPUを使用
            device = torch.device("cpu")
            print(f"Process {i}: Using CPU as MPS is limited to a single process")
        else:
            # CPU処理
            device = torch.device("cpu")

        pool.apply_async(
            inference_video,
            args=(args, sub_video_save_path, device, num_process, i),
            callback=lambda arg: pbar.update(1),
        )
    pool.close()
    pool.join()

    # combine sub videos
    # prepare vidlist.txt
    with open(f"{args.output}/{args.video_name}_vidlist.txt", "w") as f:
        for i in range(num_process):
            f.write(f"file '{args.video_name}_out_tmp_videos/{i:03d}.mp4'\n")

    cmd = [
        args.ffmpeg_bin,
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        f"{args.output}/{args.video_name}_vidlist.txt",
        "-c",
        "copy",
        f"{video_save_path}",
    ]
    print("処理済み動画を結合しています...")
    print("コマンド: " + " ".join(cmd))
    subprocess.call(cmd)
    print("出力動画を保存しました: " + video_save_path)

    # --- mkv入力時の全トラック保持mkv再パッケージ処理 ---
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

    # 一時ファイル削除
    print("一時ファイルを削除しています...")
    shutil.rmtree(osp.join(args.output, f"{args.video_name}_out_tmp_videos"))
    if osp.exists(osp.join(args.output, f"{args.video_name}_inp_tmp_videos")):
        shutil.rmtree(osp.join(args.output, f"{args.video_name}_inp_tmp_videos"))
    os.remove(f"{args.output}/{args.video_name}_vidlist.txt")
    return


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
        default="h264",
        choices=["avc", "hevc"],
        help="Video codec for output. h264 is more compatible, hevc (H.265) has better compression but slower encoding (出力動画のコーデック。h264は互換性が高く、hevc (H.265)は圧縮率が高いが処理が遅い)",
    )
    parser.add_argument(
        "--deinterlace",
        type=str,
        default="",
        choices=["", "yadif", "bwdif", "w3fdif"],
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

    # FLVファイルをMP4に変換
    if is_video and args.input.endswith(".flv"):
        mp4_path = args.input.replace(".flv", ".mp4")
        os.system(f"ffmpeg -i {args.input} -codec copy {mp4_path}")
        args.input = mp4_path

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
