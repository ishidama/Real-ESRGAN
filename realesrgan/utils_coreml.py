import platform
from typing import Optional

import cv2
import numpy as np

# CoreMLのインポート（条件付き）
try:
    import coremltools as ct  # type: ignore

    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    ct = None  # type: ignore

__all__ = ["RealESRGANerCoreML", "COREML_AVAILABLE"]


def is_apple_silicon() -> bool:
    """Apple Siliconデバイスかどうかを判定する"""
    return platform.machine() == "arm64" and platform.system() == "Darwin"


class RealESRGANerCoreML:
    """
    CoreML版のRealESRGANer。numpy配列ベースで効率的な画像処理を提供。

    固定解像度のCoreMLモデルを使用し、大きな画像はタイル分割で処理する。
    入力画像が固定解像度より小さい場合はパディングし、大きい場合はタイル分割する。
    """

    def __init__(
        self,
        model_path: str,
        scale: int = 4,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 10,
        half: bool = False,
        device: Optional[str] = None,
    ):
        """
        CoreML版RealESRGANerを初期化する。

        Args:
            model_path: CoreMLモデル(.mlpackage/.mlmodel)のパス
            scale: 超解像倍率
            tile: タイルサイズ（0の場合はモデル固有のサイズを使用）
            tile_pad: タイル間のパディングサイズ
            pre_pad: 前処理パディングサイズ
            half: FP16使用フラグ（CoreMLでは自動判定）
            device: デバイス指定（CoreMLでは無視）
        """
        if not COREML_AVAILABLE:
            raise ImportError(
                "CoreMLが利用できません。coremltools をインストールしてください: "
                "pip install coremltools"
            )

        if not is_apple_silicon():
            print(
                "⚠️  警告: Apple Silicon以外のデバイスではCoreMLの性能が劣る可能性があります"
            )

        self.model_path = model_path
        self.scale = scale
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.half = half
        self.device = device

        # CoreMLモデルをロード
        print(f"CoreMLモデルをロード中: {model_path}")
        self.model = ct.models.MLModel(model_path)  # type: ignore

        # モデルの入力名を固定（デバッグ結果から判明）
        self.input_name = "input_image"  # デバッグで確認された入力名

        print(f"🔍 モデル入力情報:")
        print(f"   入力名: '{self.input_name}'")

        # 入力サイズをモデル変換時の設定に合わせて固定
        self.input_size = 256  # pytorch2coreml.pyで設定した固定サイズ
        print(f"   入力サイズ: {self.input_size}x{self.input_size}")

        # タイルサイズの設定
        if tile == 0:
            self.tile = self.input_size  # モデルの固定サイズを使用
        else:
            self.tile = min(tile, self.input_size)  # 指定サイズとモデルサイズの小さい方

        print(f"モデル入力サイズ: {self.input_size}x{self.input_size}")
        print(f"タイルサイズ: {self.tile}x{self.tile}")
        print(f"スケール倍率: {self.scale}x")

    def enhance(
        self,
        img: np.ndarray,
        outscale: Optional[float] = None,
        alpha_upsampler: str = "realesrgan",
    ) -> np.ndarray:
        """
        画像を超解像処理する。

        Args:
            img: 入力画像（BGR、uint8、形状: [H, W, 3]）
            outscale: 出力スケール（Noneの場合はself.scaleを使用）
            alpha_upsampler: アルファチャンネルの処理方法（未使用）

        Returns:
            超解像された画像（BGR、uint8、形状: [H*scale, W*scale, 3]）
        """
        if outscale is None:
            outscale = self.scale

        h, w = img.shape[:2]

        # アルファチャンネルの処理
        if img.ndim == 3 and img.shape[2] == 4:
            img, alpha = img[:, :, :3], img[:, :, 3:]
            has_alpha = True
        else:
            has_alpha = False
            alpha = None

        # タイル分割が必要かどうかを判定
        if h <= self.tile and w <= self.tile:
            # 小さい画像：パディングして単一推論
            output = self._enhance_single_tile(img)
        else:
            # 大きい画像：タイル分割して推論
            output = self._enhance_with_tiles(img)

        # アルファチャンネルの復元
        if has_alpha and alpha is not None:
            # アルファチャンネルをスケールアップ（単純な補間）
            alpha_scaled = cv2.resize(
                alpha.astype(np.uint8),
                (output.shape[1], output.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            output = np.concatenate([output, alpha_scaled], axis=2)

        # 出力スケールの調整
        if outscale != self.scale:
            output = cv2.resize(
                output.astype(np.uint8),
                (int(w * outscale), int(h * outscale)),
                interpolation=cv2.INTER_LINEAR,
            )

        return output.astype(np.uint8)

    def _enhance_single_tile(self, img: np.ndarray) -> np.ndarray:
        """
        単一タイル（モデル固定サイズ以下）の画像を処理する。
        """
        h, w = img.shape[:2]

        # パディングしてモデルサイズに合わせる
        pad_h = max(0, self.input_size - h)
        pad_w = max(0, self.input_size - w)

        if pad_h > 0 or pad_w > 0:
            # 右と下にパディング
            padded_img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        else:
            # モデルサイズにクロップ
            padded_img = img[: self.input_size, : self.input_size]

        # CoreML推論
        output = self._coreml_inference(padded_img)

        # 元のサイズに対応する部分をクロップ
        output_h = h * self.scale
        output_w = w * self.scale
        output = output[:output_h, :output_w]

        return output

    def _enhance_with_tiles(self, img: np.ndarray) -> np.ndarray:
        """
        タイル分割して大きな画像を処理する。
        """
        h, w = img.shape[:2]
        output_h, output_w = h * self.scale, w * self.scale
        output = np.zeros((output_h, output_w, 3), dtype=np.float32)

        tile_size = self.tile
        tile_pad = self.tile_pad

        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                # タイル座標の計算
                y1 = max(0, y - tile_pad)
                x1 = max(0, x - tile_pad)
                y2 = min(h, y + tile_size + tile_pad)
                x2 = min(w, x + tile_size + tile_pad)

                # タイルを切り出し
                tile_img = img[y1:y2, x1:x2]

                # タイルを処理
                tile_output = self._enhance_single_tile(tile_img)

                # 出力座標の計算
                out_y1 = y1 * self.scale
                out_x1 = x1 * self.scale
                out_y2 = min(output_h, out_y1 + tile_output.shape[0])
                out_x2 = min(output_w, out_x1 + tile_output.shape[1])

                # パディング分を考慮したクロップ
                crop_y1 = (y - y1) * self.scale if y > y1 else 0
                crop_x1 = (x - x1) * self.scale if x > x1 else 0
                crop_y2 = crop_y1 + (out_y2 - out_y1)
                crop_x2 = crop_x1 + (out_x2 - out_x1)

                # 出力にコピー
                output[out_y1:out_y2, out_x1:out_x2] = tile_output[
                    crop_y1:crop_y2, crop_x1:crop_x2
                ]

        return output.astype(np.uint8)

    def _coreml_inference(self, img: np.ndarray) -> np.ndarray:
        """
        CoreMLモデルで推論を実行する。

        Args:
            img: 入力画像（BGR、uint8、形状: [input_size, input_size, 3]）

        Returns:
            超解像された画像（BGR、uint8、形状: [input_size*scale, input_size*scale, 3]）
        """
        # CoreMLの入力形式に変換（PIL.ImageまたはCVPixelBuffer）
        # ここではPIL.Imageを使用
        from PIL import Image

        # BGR → RGB変換
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # CoreML推論
        input_dict = {self.input_name: pil_img}
        predictions = self.model.predict(input_dict)  # type: ignore

        # 出力を取得
        output_key = list(predictions.keys())[0]
        output_pil = predictions[output_key]

        # PIL.Image → numpy配列
        output_rgb = np.array(output_pil)

        # RGB → BGR変換
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

        return output_bgr
