#!/usr/bin/env python3
"""
CoreML版RealESRGANerのテストスクリプト

作成されたCoreMLモデルの動作を確認し、PyTorch版と比較する。
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan import COREML_AVAILABLE, RealESRGANer, RealESRGANerCoreML
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def load_pytorch_model(model_path: str, scale: int = 4) -> RealESRGANer:
    """PyTorch版のRealESRGANerを作成"""
    # モデルアーキテクチャの決定
    if "x4plus" in model_path:
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale,
        )
    elif "x2plus" in model_path:
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
    else:
        # デフォルト
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale,
        )

    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=256,  # CoreMLと合わせる
        tile_pad=10,
        pre_pad=10,
        half=False,  # CPUの場合はFalse
    )

    return upsampler


def test_coreml_inference(
    coreml_path: str, pytorch_path: str, input_path: str, output_dir_str: str
):
    """CoreMLとPyTorchの推論結果を比較"""

    if not COREML_AVAILABLE:
        print("❌ CoreMLが利用できません")
        return

    # 出力ディレクトリの作成
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 入力画像の読み込み
    print(f"📷 入力画像を読み込み中: {input_path}")
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"❌ 画像が読み込めません: {input_path}")
        return

    print(f"   入力サイズ: {img.shape[1]}x{img.shape[0]}")

    # CoreML版の推論
    print("\n🚀 CoreML版で推論実行中...")
    try:
        coreml_upsampler = RealESRGANerCoreML(
            model_path=coreml_path, scale=4, tile=256, tile_pad=10
        )

        start_time = time.time()
        coreml_output = coreml_upsampler.enhance(img)
        coreml_time = time.time() - start_time

        coreml_output_path = output_dir / "output_coreml.png"
        cv2.imwrite(str(coreml_output_path), coreml_output)
        print(f"   ✅ CoreML推論完了: {coreml_time:.2f}秒")
        print(f"   出力サイズ: {coreml_output.shape[1]}x{coreml_output.shape[0]}")
        print(f"   保存先: {coreml_output_path}")

    except Exception as e:
        print(f"   ❌ CoreML推論エラー: {e}")
        return

    # PyTorch版の推論
    print("\n🔥 PyTorch版で推論実行中...")
    try:
        pytorch_upsampler = load_pytorch_model(pytorch_path, scale=4)

        start_time = time.time()
        pytorch_result = pytorch_upsampler.enhance(img, outscale=4)
        pytorch_time = time.time() - start_time

        # PyTorch出力の形式を確認・修正
        print(f"   🔍 PyTorch出力形式: {type(pytorch_result)}")

        # enhanceメソッドがタプルを返す場合、最初の要素が画像
        if isinstance(pytorch_result, tuple):
            pytorch_output = pytorch_result[0]
            print(f"   🔍 タプルから画像を抽出: {type(pytorch_output)}")
        else:
            pytorch_output = pytorch_result

        print(
            f"   🔍 画像データ: shape={pytorch_output.shape}, dtype={pytorch_output.dtype}"
        )

        # numpy配列でuint8型であることを確認
        if not isinstance(pytorch_output, np.ndarray):
            pytorch_output = np.array(pytorch_output)

        if pytorch_output.dtype != np.uint8:
            # float型の場合は0-1の範囲から0-255にスケール
            if pytorch_output.dtype in [np.float32, np.float64]:
                if pytorch_output.max() <= 1.0:
                    pytorch_output = (pytorch_output * 255).astype(np.uint8)
                else:
                    pytorch_output = np.clip(pytorch_output, 0, 255).astype(np.uint8)
            else:
                pytorch_output = pytorch_output.astype(np.uint8)

        pytorch_output_path = output_dir / "output_pytorch.png"
        success = cv2.imwrite(str(pytorch_output_path), pytorch_output)
        if not success:
            raise ValueError("画像の保存に失敗しました")

        print(f"   ✅ PyTorch推論完了: {pytorch_time:.2f}秒")
        print(f"   出力サイズ: {pytorch_output.shape[1]}x{pytorch_output.shape[0]}")
        print(f"   保存先: {pytorch_output_path}")

    except Exception as e:
        print(f"   ❌ PyTorch推論エラー: {e}")
        import traceback

        traceback.print_exc()
        return

    # 性能比較
    print(f"\n📊 性能比較:")
    print(f"   CoreML:  {coreml_time:.2f}秒")
    print(f"   PyTorch: {pytorch_time:.2f}秒")
    speedup = pytorch_time / coreml_time
    print(f"   スピードアップ: {speedup:.2f}x")

    # 画質比較（簡易）
    print(f"\n🔍 画質比較:")
    # 同じサイズにリサイズ
    min_h = min(coreml_output.shape[0], pytorch_output.shape[0])
    min_w = min(coreml_output.shape[1], pytorch_output.shape[1])

    coreml_resized = cv2.resize(coreml_output, (min_w, min_h))
    pytorch_resized = cv2.resize(pytorch_output, (min_w, min_h))

    # MSE計算
    mse = np.mean(
        (coreml_resized.astype(np.float32) - pytorch_resized.astype(np.float32)) ** 2
    )
    print(f"   MSE: {mse:.2f}")

    # PSNR計算
    if mse > 0:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        print(f"   PSNR: {psnr:.2f} dB")
    else:
        print(f"   PSNR: ∞ (完全一致)")

    # 差分画像を保存
    diff_img = np.abs(
        coreml_resized.astype(np.int16) - pytorch_resized.astype(np.int16)
    )
    diff_img = np.clip(diff_img * 10, 0, 255).astype(np.uint8)  # 差分を10倍して見やすく
    diff_output_path = output_dir / "difference.png"
    cv2.imwrite(str(diff_output_path), diff_img)
    print(f"   差分画像: {diff_output_path}")


def main():
    parser = argparse.ArgumentParser(description="CoreML版RealESRGANerのテスト")
    parser.add_argument("--coreml", required=True, help="CoreMLモデルのパス")
    parser.add_argument("--pytorch", required=True, help="PyTorchモデルのパス")
    parser.add_argument("--input", required=True, help="入力画像のパス")
    parser.add_argument("--output", default="test_results", help="出力ディレクトリ")

    args = parser.parse_args()

    print("🧪 CoreML vs PyTorch 比較テスト")
    print("=" * 50)

    test_coreml_inference(
        coreml_path=args.coreml,
        pytorch_path=args.pytorch,
        input_path=args.input,
        output_dir_str=args.output,
    )


if __name__ == "__main__":
    main()
