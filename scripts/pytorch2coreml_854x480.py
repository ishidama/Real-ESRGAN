#!/usr/bin/env python3
"""
854×480解像度専用 PyTorch RealESRGAN → CoreML 変換ツール
DVD 16:9解像度(854×480)に特化した固定解像度モデルを生成

【特徴・設計方針】
- 入力: BGR画像 (uint8, 0-255, 854×480固定) → CoreML側で0-1正規化
- 出力: BGR画像 (float, 0-255, 入力のscale倍解像度)
- 固定解像度でANE（Apple Neural Engine）の発動を保証
- 入出力ともBGR順、バッチ1枚
- 出力はtorch.clampで0-1にクリップ後、255倍

【利用例】
python scripts/pytorch2coreml_854x480.py --pth weights/RealESRGAN_x4plus.pth --out weights/RealESRGAN_x4plus_854x480.mlpackage --float16
"""

import argparse
import coremltools as ct
import sys
import torch
import traceback
from basicsr.archs.rrdbnet_arch import RRDBNet
from pathlib import Path
from typing import Optional


class ClampedModel(torch.nn.Module):
    """
    PyTorchモデルの出力を0〜1にclampし ×255 で 0〜255 レンジへ変換するラッパー。
    CoreML変換時に値域外の出力を防ぐため必須。
    """

    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        モデルのフォワードパス。

        引数:
            x (torch.Tensor): 入力テンソル。

        戻り値:
            torch.Tensor: 0から1の範囲にクランプされた後、255.0倍されたテンソル。

        注意:
            torch.clampは入力テンソルの全要素を[0.0, 1.0]の範囲内に制限します。
            0.0未満の値は0.0に、1.0より大きい値は1.0に設定されます。
            これにより、255を掛けた後も画像表現として有効な出力値であることが保証されます。
        """
        out = self.base_model(x)
        return torch.clamp(out, 0.0, 1.0) * 255.0


def load_rrdbnet(pth_path: str, scale: int = 4) -> torch.nn.Module:
    """
    RealESRGAN (RRDBNet) モデルを構築し、重みをロードする。
    - 入出力: BGR, float32, 0-1, (N,3,H,W)

    Args:
        pth_path: PyTorch重みファイルのパス
        scale: スケール倍率（デフォルト: 4）
    """
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale,
    )
    ckpt = torch.load(pth_path, map_location="cpu")
    for key in ("params_ema", "params"):
        if key in ckpt:
            model.load_state_dict(ckpt[key], strict=True)
            break
    else:
        raise RuntimeError(
            f"'{pth_path}'ファイル内にparams_emaまたはparamsキーが見つかりません"
        )
    model.eval()
    return model


def validate_args(args: argparse.Namespace) -> Optional[str]:
    """引数の検証を行う"""
    if args.scale not in [2, 4]:
        return f"スケールは2または4である必要があります（指定値: {args.scale}）"
    if getattr(args, "trace_size", 32) <= 0 or getattr(args, "trace_size", 32) % 4:
        return "--trace-size は正の4の倍数にしてください"
    return None


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch RealESRGAN (.pth) → CoreML 変換ツール (固定解像度版)"
    )
    parser.add_argument("--pth", required=True, help="PyTorch重みファイルのパス")
    parser.add_argument("--out", required=True, help="出力CoreMLモデルのパス")
    # 入力サイズは固定なのでパラメーターは不要
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        choices=[2, 4],
        help="超解像倍率（デフォルト: 4）",
    )
    parser.add_argument(
        "--float16", action="store_true", help="FP16精度で変換 (省略時はFP32)"
    )
    parser.add_argument(
        "--trace-size",
        type=int,
        default=32,
        help="トレース時に使うダミー入力の一辺 (デフォルト: 32)。"
             "モデル変換時の実際の入出力形状とは無関係なので小さいほうが省メモリ。"
    )
    parser.add_argument(
        "--target",
        choices=["mac", "ios"],
        default="mac",
        help="デプロイターゲット(mac/ios)",
    )
    parser.add_argument(
        "--compute-units",
        choices=["all", "cpu", "ane", "gpu"],
        default="ane",
        help="計算ユニット (all: CPU+GPU+ANE, cpu: CPUのみ, ane: CPU+ANE, gpu: CPU+GPU) (default: ane)",
    )
    args = parser.parse_args()

    # 引数のバリデーション
    error_msg = validate_args(args)
    if error_msg:
        print(f"❌ 引数エラー: {error_msg}", file=sys.stderr)
        sys.exit(1)

    # 出力ディレクトリの作成
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 1. モデル構築・重みロード
        print(f"[INFO] PyTorchモデルをロード中: {args.pth}")
        model = load_rrdbnet(args.pth, args.scale)
        model = ClampedModel(model)  # 出力を0-1にclampするラッパー

        # 2. 入力・出力仕様を定義（854×480固定解像度）
        # 入力サイズは引数を無視して854×480に固定
        width = 854
        height = 480
        output_width = width * args.scale
        output_height = height * args.scale

        input_desc = ct.ImageType(
            name="input_image",
            shape=(1, 3, height, width),  # 固定解像度: 854×480
            color_layout=ct.colorlayout.BGR,
            bias=[0.0, 0.0, 0.0],
            scale=1 / 255.0,  # uint8→float32(0-1)
        )

        output_desc = ct.ImageType(
            name="output_image",
            color_layout=ct.colorlayout.BGR,
        )

        # 3. 変換オプション
        precision = ct.precision.FLOAT16 if args.float16 else ct.precision.FLOAT32
        target_map = {"mac": ct.target.macOS14, "ios": ct.target.iOS17}

        # compute_unitsのマッピング
        compute_units_map = {
            "all": ct.ComputeUnit.ALL,
            "cpu": ct.ComputeUnit.CPU_ONLY,
            "ane": ct.ComputeUnit.CPU_AND_NE,  # ANE（Neural Engine）使用
            "gpu": ct.ComputeUnit.CPU_AND_GPU,
        }

        conversion_options = {
            "source": "pytorch",
            "inputs": [input_desc],
            "outputs": [output_desc],
            "convert_to": "mlprogram",
            "minimum_deployment_target": target_map[args.target],
            "compute_precision": precision,
            "compute_units": compute_units_map[args.compute_units],
        }

        # 4. トレース用ダミー入力（省メモリ）
        trace_h = trace_w = args.trace_size
        example_input = torch.zeros(1, 3, trace_h, trace_w, dtype=torch.float32)
        print(f"[INFO] モデルをトレース中: ダミー入力 {trace_w}x{trace_h}")
        traced = torch.jit.trace(model, example_input)

        # 5. CoreML変換
        print("[INFO] 入力: BGR, uint8, 0-255 → CoreMLで0-1正規化 (scale=1/255.0)")
        print("[INFO] 出力: BGR, float32, 0-255 (ImageType で即 PNG/JPEG 保存可能)")
        print(f"[INFO] 入力shape: (1,3,{height},{width}) 固定解像度")
        print(f"[INFO] 出力shape: (1,3,{output_height},{output_width}) 固定解像度")
        print(f"[INFO] スケール倍率: {args.scale}x")
        print(
            f"[INFO] 計算ユニット: {args.compute_units} ({compute_units_map[args.compute_units]})"
        )
        print(f"[INFO] 精度: {'FP16' if args.float16 else 'FP32'}")
        print("[INFO] ANE発動のため固定解像度で変換中...")

        mlmodel = ct.convert(traced, **conversion_options)

        # 6. CoreMLモデルの保存
        print(f"[INFO] CoreMLモデルを保存中: {args.out}")
        # MLProgramタイプに対するcoremltools 8.0以降の保存方法
        mlmodel.save(args.out)

        print("[INFO] 変換完了")
        print(f"[INFO] 保存先: {output_path.absolute()}")
        print(f"[INFO] 入力解像度: {width}x{height}")
        print(f"[INFO] 出力解像度: {output_width}x{output_height}")

    except Exception as e:
        print(f"❌ 変換エラー: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
