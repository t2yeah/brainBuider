**音から空間を理解する生成AI：ヤドリギ ブレインビルダー
1. プロジェクト概要

本プロジェクトは、マイクが捉えた「音」からその場の「空間・状況・物語」を解釈し、直感的に理解しやすい**1コマ漫画とオノマトペ（擬音語）**として可視化するシステムです。

環境音から「どこで」「何が」起きているかをAIが多角的に分析し、単なる音のタグ付けを超えた「情景の視覚化」を提供します。

主な機能 / ユースケース

環境音の空間・イベント解析: PANNsとCLAPを用い、音の種類と空間属性を特定。

多段階LLMによる情景解釈: 音の文脈から「物語」を生成し、画像生成用の高度な指示書（プロンプト）を構築。

AI漫画生成: Animagine XL 4.0を使用し、物語に沿った1コマ漫画を生成。

動的オノマトペ合成: 音の物理特徴（振幅や勢い）を解析し、フォントサイズや角度を調整して描き文字を合成。

ユースケース: 防犯カメラの視覚化、聴覚障がい者向けの情報保障、音響データのインタラクティブなアーカイブなど。
2. 技術スタック

言語: Python 3.10+

音声処理: librosa, panns-inference, LAION-CLAP

LLM（推論）: transformers, accelerate, bitsandbytes (8-bit quantization)

画像生成: diffusers (Stable Diffusion XL)

画像加工: Pillow (PIL)

Web API: FastAPI, uvicorn
3. モデル情報

本パイプラインでは以下のモデルを使用しています。

カテゴリ
音声イベント解析
空間属性推定
日本語LLM
画像生成
4. 使用データ

検証用データ: ハッカソン提供の環境音データ、および data/uploads に配置された任意の音声ファイル（wav, mp3, m4a等）。

空間定義: config/space_taxonomy.yaml に定義された空間カテゴリ。
5. セットアップ手順
前提環境

OS: Ubuntu 22.04 LTS

Python: 3.10 以上

GPU: NVIDIA RTX A5000 (24GB VRAM) 以上を強く推奨

依存ツール: ffmpeg (音声変換用)
インストール
Bash
# リポジトリのクローン
git clone https://github.com/your-team/brain-builder.git
cd brain-builder
# システム依存のインストール
sudo apt update && sudo apt install -y ffmpeg
# Pythonライブラリのインストール
pip install -r requirements.txt
環境変数設定

.env ファイルを作成し、以下の情報を設定してください。

Bash
HF_TOKEN=your_huggingface_token_here  # Swallow/Animagineの取得に必要
SWALLOW_LOCAL_ENABLED=1
MANGA_PROMPT_LLM=1
6. 実行方法（推論）
エントリーポイント

全ての処理を統合する pipeline.py を実行します。

Bash
# data/uploads/example.wav がある場合
python pipeline.py --audio-id example
処理フロー

音声前処理: audio_preprocess.py (FFmpegによる正規化・分割)

特徴抽出: audio_features.py & audio_analyze.py (PANNs等)

情景解釈: agent_scene_interpreter.py (Swallow-8B)

画像生成: generate_manga_image.py (Animagine XL)

合成・統合: manga_text.py & final_result.py
7. ディレクトリ構成
project
.
├── app/
│   └── services/       # 各推論・処理エンジンの実装
├── config/             # YAML設定（空間定義、推論ルール等）
├── data/
│   ├── uploads/        # 入力音声配置
│   ├── segments/       # 中間処理（分割済み音声）
│   └── results/        # 最終出力（JSON、画像）
├── fonts/              # 合成用フォント
├── pipeline.py         # メインエントリーポイント
└── requirements.txt    # 依存ライブラリ
8. 制約・注意事項

計算資源: 本システムは VRAM 24GB 前後の環境を想定しています。推論ごとに torch.cuda.empty_cache() を呼び出していますが、複数の巨大モデルをロードするため、GPUメモリの競合にご注意ください。

ネットワーク: 初回実行時は Hugging Face から数GB〜十数GBのモデルデータをダウンロードします。

外部依存: 音声解析に FFmpeg を使用しているため、パスが通っている必要があります。
9. ライセンス

本プロジェクトのソースコードは MIT License です。

ただし、使用している各AIモデル（Llama-3.1-Swallow, Animagine XL等）の利用に関しては、それぞれのモデルのライセンス規定に従ってください。
