#!/usr/bin/env python3
"""
CoreMLモデルの詳細情報を表示するデバッグスクリプト
"""

import sys

import coremltools as ct


def debug_coreml_model(model_path: str):
    """CoreMLモデルの詳細情報を表示する"""
    print(f"🔍 CoreMLモデル調査: {model_path}")
    print("=" * 60)

    try:
        # モデルをロード
        model = ct.models.MLModel(model_path)

        # 入力仕様の詳細
        print("📥 入力仕様:")
        input_spec = model.input_description
        print(f"   入力数: {len(input_spec)}")

        for i, inp in enumerate(input_spec):
            print(f"\n   入力 {i}:")
            print(f"     名前: '{inp}'")
            print(f"     タイプ: {type(inp)}")

            # input_specがdict形式の場合
            if isinstance(input_spec, dict):
                inp_name = inp
                inp_desc = input_spec[inp]
                print(f"     実際の名前: '{inp_name}'")
                print(f"     説明: {inp_desc}")
                print(f"     説明タイプ: {type(inp_desc)}")

                # Image型の場合
                if hasattr(inp_desc.type, "imageType"):
                    img_type = inp_desc.type.imageType
                    print(f"     画像高さ: {img_type.height}")
                    print(f"     画像幅: {img_type.width}")
                    print(f"     カラーレイアウト: {img_type.colorSpace}")

                # MultiArray型の場合
                elif hasattr(inp_desc.type, "multiArrayType"):
                    array_type = inp_desc.type.multiArrayType
                    print(f"     形状: {array_type.shape}")
                    print(f"     データ型: {array_type.dataType}")
            else:
                # リスト形式の場合
                if hasattr(inp, "name"):
                    print(f"     実際の名前: '{inp.name}'")

                if hasattr(inp, "type"):
                    print(f"     タイプ詳細: {inp.type}")

                    # Image型の場合
                    if hasattr(inp.type, "imageType"):
                        img_type = inp.type.imageType
                        print(f"     画像高さ: {img_type.height}")
                        print(f"     画像幅: {img_type.width}")
                        print(f"     カラーレイアウト: {img_type.colorSpace}")

                    # MultiArray型の場合
                    elif hasattr(inp.type, "multiArrayType"):
                        array_type = inp.type.multiArrayType
                        print(f"     形状: {array_type.shape}")
                        print(f"     データ型: {array_type.dataType}")

        # 出力仕様の詳細
        print("\n📤 出力仕様:")
        output_spec = model.output_description
        print(f"   出力数: {len(output_spec)}")

        for i, out in enumerate(output_spec):
            print(f"\n   出力 {i}:")
            print(f"     名前: '{out}'")
            print(f"     タイプ: {type(out)}")

            # output_specがdict形式の場合
            if isinstance(output_spec, dict):
                out_name = out
                out_desc = output_spec[out]
                print(f"     実際の名前: '{out_name}'")
                print(f"     説明: {out_desc}")
                print(f"     説明タイプ: {type(out_desc)}")

                # Image型の場合
                if hasattr(out_desc.type, "imageType"):
                    img_type = out_desc.type.imageType
                    print(f"     画像高さ: {img_type.height}")
                    print(f"     画像幅: {img_type.width}")
                    print(f"     カラーレイアウト: {img_type.colorSpace}")

                # MultiArray型の場合
                elif hasattr(out_desc.type, "multiArrayType"):
                    array_type = out_desc.type.multiArrayType
                    print(f"     形状: {array_type.shape}")
                    print(f"     データ型: {array_type.dataType}")
            else:
                # リスト形式の場合
                if hasattr(out, "name"):
                    print(f"     実際の名前: '{out.name}'")

                if hasattr(out, "type"):
                    print(f"     タイプ詳細: {out.type}")

                    # Image型の場合
                    if hasattr(out.type, "imageType"):
                        img_type = out.type.imageType
                        print(f"     画像高さ: {img_type.height}")
                        print(f"     画像幅: {img_type.width}")
                        print(f"     カラーレイアウト: {img_type.colorSpace}")

                    # MultiArray型の場合
                    elif hasattr(out.type, "multiArrayType"):
                        array_type = out.type.multiArrayType
                        print(f"     形状: {array_type.shape}")
                        print(f"     データ型: {array_type.dataType}")

        # メタデータ
        print("\n📋 メタデータ:")
        metadata = model.user_defined_metadata or {}
        if metadata:
            for key, value in metadata.items():
                print(f"   {key}: {value}")
        else:
            print("   メタデータなし")

        print("\n✅ モデル調査完了")

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用法: python debug_coreml_model.py <model_path>")
        sys.exit(1)

    debug_coreml_model(sys.argv[1])
