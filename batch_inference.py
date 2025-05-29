#!/usr/bin/env python3
# filepath: /Users/ishimada/Projects/Real-ESRGAN/batch_inference.py
"""
バッチ推論スクリプト for Real-ESRGAN

CSVファイルで指定されたレシピに基づいて、複数の動画ファイルに対してReal-ESRGANを実行します。
進捗状況を表示し、処理の成功/失敗を記録します。
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def parse_args():
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(
        description="バッチ処理で複数のファイルにReal-ESRGANを適用します"
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default="inference_recipe.csv",
        help="処理レシピを記述したCSVファイル（デフォルト: inference_recipe.csv）",
    )
    parser.add_argument(
        "--ffmpeg_bin",
        type=str,
        default="ffmpeg",
        help="ffmpegのパス（デフォルト: 'ffmpeg'）",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="使用するGPUのID（デフォルト: 0）"
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=0,
        help="タイルサイズ。メモリ不足エラーが出る場合は400程度を指定（デフォルト: 0）",
    )
    parser.add_argument(
        "--deinterlace",
        type=str,
        default="auto",
        choices=["auto", "", "yadif", "bwdif", "w3fdif"],
        help="デインターレースフィルター（デフォルト: auto）。"
        "レシピCSVファイルにdeinterlace列が存在する場合はそちらが優先されます。",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="hevc",
        choices=["h264", "hevc"],
        help="出力コーデック。hevcはH.265で高画質だがエンコードが遅い（デフォルト: hevc）",
    )
    parser.add_argument(
        "--outscale", type=float, default=4, help="拡大倍率（デフォルト: 4）"
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="FP32精度で実行（通常はFP16、Apple Siliconでは精度向上のためFP32推奨）",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="既に出力ファイルが存在する場合はスキップする",
    )
    return parser.parse_args()


def read_recipe(csv_path: str) -> List[Dict[str, str]]:
    """CSVファイルからレシピ情報を読み込む"""
    recipes = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # コメント行（#で始まる）をスキップ
                if row["input_folder"].startswith("#"):
                    continue
                recipes.append(row)
        return recipes
    except Exception as e:
        print(f"エラー: CSVファイル '{csv_path}' の読み込みに失敗しました: {e}")
        sys.exit(1)


def format_time(seconds: float) -> str:
    """経過時間を見やすい形式にフォーマットする"""
    td = timedelta(seconds=seconds)
    if td.days > 0:
        return f"{td.days}日 {td.seconds // 3600:02}:{(td.seconds // 60) % 60:02}:{td.seconds % 60:02}"
    else:
        return (
            f"{td.seconds // 3600:02}:{(td.seconds // 60) % 60:02}:{td.seconds % 60:02}"
        )


def run_inference(recipe: Dict[str, str], args) -> Tuple[bool, str]:
    """1つのレシピに対して推論処理を実行する"""
    input_folder = recipe["input_folder"]
    input_file = recipe["input_file"]
    output_folder = recipe["output_folder"]
    output_suffix = recipe["output_suffix"]
    model_name = recipe["model_name"]
    denoise_strength = recipe.get("denoise_strength", "0.5")  # デフォルト値

    # 入力ファイルパスと出力ファイル名の構築
    input_path = os.path.join(input_folder, input_file)
    if not os.path.exists(input_path):
        return False, f"入力ファイルが見つかりません: {input_path}"

    # 出力フォルダが存在しない場合は作成
    os.makedirs(output_folder, exist_ok=True)

    # 出力ファイル名の拡張子を確認してMKVに統一
    filename_without_ext = os.path.splitext(input_file)[0]
    output_file = f"{filename_without_ext}_{output_suffix}.mkv"
    output_path = os.path.join(output_folder, output_file)

    # 既に出力ファイルが存在する場合はスキップする機能
    if args.skip_existing and os.path.exists(output_path):
        return True, f"出力ファイルが既に存在するためスキップします: {output_path}"

    # inference_realesrgan_video.py に渡す引数の構築
    cmd = [
        sys.executable,
        "inference_realesrgan_video.py",
        "-i",
        input_path,
        "-o",
        output_folder,
        "-n",
        model_name,
        "--suffix",
        output_suffix,
        "-dn",
        denoise_strength,
    ]

    # デインタレース設定（レシピ優先、次にコマンドライン引数）
    deinterlace_value = recipe.get("deinterlace", "").strip()
    if deinterlace_value and deinterlace_value in ["yadif", "bwdif", "w3fdif"]:
        # レシピでデインタレースが指定されている場合
        cmd.extend(["--deinterlace", deinterlace_value])
        print(f"デインタレース設定: {deinterlace_value} (レシピから)")
    elif args.deinterlace and args.deinterlace not in ["", "auto"]:
        # レシピに無くてもコマンドラインで有効な設定がある場合
        cmd.extend(["--deinterlace", args.deinterlace])
        print(f"デインタレース設定: {args.deinterlace} (コマンドラインから)")
    elif args.deinterlace == "auto":
        # 自動検出を使用
        cmd.extend(["--deinterlace", "auto"])
        print("デインタレース設定: 自動検出")
    else:
        print("デインタレース設定: なし")

    # その他のコマンドラインオプション
    # タイルサイズの設定
    if args.tile > 0:
        cmd.extend(["-t", str(args.tile)])
        print(f"タイルサイズ: {args.tile}")

    # コーデック設定
    if args.codec:
        cmd.extend(["--codec", args.codec])
        print(f"出力コーデック: {args.codec}")

    # 拡大倍率
    if args.outscale != 4:  # デフォルト値と異なる場合のみ指定
        cmd.extend(["-s", str(args.outscale)])
        print(f"拡大倍率: {args.outscale}倍")

    # FP32精度
    if args.fp32:
        cmd.append("--fp32")
        print("FP32精度: 有効")

    # コマンド実行（サブプロセスにリストとして渡すので、スペース含むファイル名も安全に扱える）
    cmd_str = " ".join(f"'{arg}'" if " " in str(arg) else str(arg) for arg in cmd)
    print(f"\n実行コマンド: {cmd_str}\n")

    try:
        start_time = time.time()
        # キャプチャせずに標準出力・標準エラー出力を直接表示させる
        subprocess.run(cmd, check=True, capture_output=False, text=True)
        processing_time = time.time() - start_time

        # 成功
        return True, f"処理時間: {format_time(processing_time)}"
    except subprocess.CalledProcessError:
        return False, "実行エラーが発生しました"
    except Exception as e:
        return False, f"予期せぬエラー: {str(e)}"


def main():
    """メイン処理"""
    args = parse_args()

    # レシピファイルの読み込み
    print(f"レシピファイル '{args.recipe}' を読み込んでいます...")
    recipes = read_recipe(args.recipe)
    if not recipes:
        print("処理対象のレシピがありません。")
        return

    total_count = len(recipes)
    print(f"処理対象: {total_count} 件のファイルが見つかりました。\n")

    # 処理結果記録用
    results = []
    start_total = time.time()

    # 各レシピを順番に処理
    for i, recipe in enumerate(recipes, 1):
        print(f"【処理 {i}/{total_count}】")
        print(f"入力: {os.path.join(recipe['input_folder'], recipe['input_file'])}")
        print(
            f"出力: {recipe['output_folder']}/{os.path.splitext(recipe['input_file'])[0]}_{recipe['output_suffix']}.mkv"
        )
        print(f"モデル: {recipe['model_name']}")
        print(f"ノイズ除去強度: {recipe['denoise_strength']}")

        # デインタレース設定の表示
        deinterlace = recipe.get("deinterlace", "").strip()
        if deinterlace:
            print(f"デインタレース: {deinterlace}")
        else:
            print("デインタレース: 設定なし")

        # 処理実行
        print("処理中...")
        success, message = run_inference(recipe, args)

        # 結果表示
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"{status}: {message}")

        # 結果を記録
        results.append(
            {
                "index": i,
                "input": os.path.join(recipe["input_folder"], recipe["input_file"]),
                "status": status,
                "message": message,
            }
        )

    # 処理結果サマリーの表示
    total_time = time.time() - start_total
    success_count = sum(1 for r in results if "成功" in r["status"])

    print("\n" + "=" * 60)
    print(f"処理完了: 合計 {total_count} 件")
    print(f"成功: {success_count} 件 / 失敗: {total_count - success_count} 件")
    print(f"総処理時間: {format_time(total_time)}")
    print("=" * 60)

    # 詳細結果の表示
    print("\n詳細結果:")
    for r in results:
        print(
            f"{r['index']}. {r['status']} - {os.path.basename(r['input'])} - {r['message']}"
        )


if __name__ == "__main__":
    main()
