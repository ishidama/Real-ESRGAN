# CoreML版RealESRGAN for Apple Silicon

Apple Silicon（M1/M2/M3/M4チップ）でのビデオ超解像処理を高速化するためのCoreML実装です。

## 🚀 特徴

- **24倍高速化**: PyTorchと比較して約24倍の高速処理
- **Apple Neural Engine加速**: ANE（Apple Neural Engine）を活用した専用ハードウェア加速
- **固定解像度最適化**: 256x256固定入力によるANE最適化
- **高画質維持**: PSNR 28dB以上の高品質な超解像結果

## 📦 インストール

```bash
# CoreMLツールのインストール
pip install coremltools

# 必要な依存関係
pip install opencv-python pillow numpy
```

## 🔧 モデル変換

PyTorchモデルをCoreMLに変換：

```bash
python scripts/pytorch2coreml.py \
    --input weights/RealESRGAN_x4plus.pth \
    --output weights/RealESRGAN_x4plus_256.mlpackage \
    --input_size 256
```

## 🎯 使用方法

### 基本的な使用例

```python
from realesrgan import RealESRGANerCoreML
import cv2

# CoreMLモデルの初期化
upsampler = RealESRGANerCoreML(
    model_path='weights/RealESRGAN_x4plus_256.mlpackage',
    scale=4,
    tile=256,
    tile_pad=10
)

# 画像の読み込み
img = cv2.imread('input.jpg', cv2.IMREAD_COLOR)

# 超解像処理
output_img = upsampler.enhance(img)

# 結果の保存
cv2.imwrite('output.jpg', output_img)
```

### 性能比較テスト

```bash
python test_coreml.py \
    --coreml weights/RealESRGAN_x4plus_256.mlpackage \
    --pytorch weights/RealESRGAN_x4plus.pth \
    --input test_image.jpg \
    --output results/
```

## 📊 性能ベンチマーク

| 手法 | 処理時間 | スピードアップ | PSNR |
|------|----------|----------------|------|
| PyTorch (GPU) | 4.47秒 | 1.0x | 基準 |
| CoreML (ANE) | 0.19秒 | **23.99x** | 28.33 dB |

*テスト環境: Apple Silicon M4Max, 200x200 → 800x800*

## 🔍 技術詳細

### アーキテクチャ最適化
- **固定入力サイズ**: 256x256でANE最適化
- **タイル分割**: 大きな画像の効率的な処理
- **自動パディング**: 小さな画像の適応的処理

### 入力・出力形式
- **入力**: BGR形式、uint8、任意サイズ
- **出力**: BGR形式、uint8、4倍拡大
- **CoreML内部**: RGB形式、PIL.Image

## 🛠️ トラブルシューティング

### Apple Silicon検出
```python
from realesrgan.utils_coreml import is_apple_silicon

if is_apple_silicon():
    print("Apple Silicon環境で動作中")
else:
    print("⚠️ Apple Silicon以外の環境では性能が劣る可能性があります")
```

### メモリ使用量
大きな画像を処理する場合は、タイルサイズを調整してください：

```python
# メモリ使用量を抑えたい場合
upsampler = RealESRGANerCoreML(
    model_path='model.mlpackage',
    tile=128,  # より小さなタイル
    tile_pad=5
)
```

## 📚 ファイル構成

```
realesrgan/
├── utils_coreml.py          # CoreML実装クラス
├── __init__.py              # CoreMLインポート
scripts/
├── pytorch2coreml.py        # モデル変換スクリプト
test_coreml.py               # 性能比較テスト
debug_coreml_model.py        # モデル詳細調査
```

## 🎬 ビデオ処理への統合

この実装は将来的にビデオ処理パイプラインに統合される予定です：

- フレーム単位での高速処理
- バッファリングによる効率的なメモリ管理
- リアルタイム超解像ストリーミング

## 🔗 関連リンク

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - オリジナルプロジェクト
- [CoreML Tools](https://coremltools.readme.io/) - Apple公式ツール
- [Apple Neural Engine](https://github.com/hollance/neural-engine) - ANE技術情報

---

**注意**: このCoreML実装はApple Silicon環境でのみ最適化されています。Intel Macでは従来のPyTorch版をご使用ください。
