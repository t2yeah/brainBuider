Sound Space Analyzer: ヤドリギ ブレインビルダー		
1. プロジェクト概要		
		
本プロジェクトは、マイクが捉えた「音」からその場の「空間・状況・物語」を解釈し、直感的に理解しやすい**1コマ漫画とオノマトペ（擬音語）**として可視化するシステムです。		
		
環境音から「どこで」「何が」起きているかをAIが多角的に分析し、単なる音のタグ付けを超えた「情景の視覚化」を提供します。		
		
主な機能 / ユースケース		
		
環境音の空間・イベント解析: PANNsとCLAPを用い、音の種類と空間属性を特定。		
		
多段階LLMによる情景解釈: 音の文脈から「物語」を生成し、画像生成用の高度な指示書（プロンプト）を構築。		
		
AI漫画生成: Animagine XL 4.0を使用し、物語に沿った1コマ漫画を生成。		
		
動的オノマトペ合成: 音の物理特徴（振幅や勢い）を解析し、フォントサイズや角度を調整して描き文字を合成。		
		
Web ダッシュボード: 解析結果をブラウザ上でインタラクティブに確認可能（index.html）。		
		
ユースケース: 防犯カメラの視覚化、聴覚障がい者向けの情報保障、音響データのインタラクティブなアーカイブなど。		
2. 技術スタック		
		
言語: Python 3.10+		
		
音声処理: librosa, panns-inference, LAION-CLAP		
		
LLM（推論）: Transformers (Llama-3.1-Swallow-8B-Instruct-v0.3), accelerate, bitsandbytes (8-bit quantization)		
		
画像生成: diffusers (Animagine XL 4.0)		
		
Web API/Front: FastAPI, uvicorn, HTML5, Vanilla JS, ngrok (外部トンネリング)		
		
インフラ: Slurm (ジョブ管理), Enroot (コンテナ実行)		
3. モデル情報		
		
本パイプラインでは以下のモデルを使用しています。		
		
カテゴリ	モデル名	役割
音声イベント解析	PANNs (Cnn14)	音の種類（タグ）の特定
空間属性推定	LAION-CLAP	空間カテゴリとの親和性計算
日本語LLM	Llama-3.1-Swallow-8B	シーン解釈・プロンプト生成
画像生成	Animagine XL 4.0	1コマ漫画（画像）の生成
4. セットアップ手順		
前提環境		
		
OS: Ubuntu 22.04 LTS		
		
GPU: NVIDIA RTX A5000 (24GB VRAM) 以上を強く推奨（H100対応）		
		
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
		
.env ファイルを作成するか、シェルにエクスポートしてください。		
Bash		
HF_TOKEN=your_huggingface_token_here  # モデル取得に必須		
SWALLOW_LOCAL_ENABLED=1		
MANGA_PROMPT_LLM=1		
NGROK_AUTHTOKEN=your_ngrok_token      # Web公開時に使用		
5. 実行方法		
A. 解析パイプラインの実行（CLI）		
		
全ての処理を統合する pipeline.py を実行します。		
Bash		
# data/uploads/example.wav を解析する場合		
python pipeline.py --audio-id example		
		
処理フロー:		
		
audio_preprocess.py: 音声の正規化・分割		
		
audio_features.py & audio_analyze.py: 特徴抽出・イベント解析		
		
agent_scene_interpreter.py: LLMによる情景解釈		
		
generate_manga_image.py: AI漫画生成		
		
manga_text.py & final_result.py: オノマトペ合成・結果統合		
B. Webサーバーの起動（Slurm環境）		
		
計算資源を確保しながら Web サーバーを立ち上げるためのスクリプト群です。		
		
Bash		
cd project/site		
# サーバーの起動 (Slurmジョブとして発行)		
bash start_web.sh		
# 起動ステータス・URLの確認		
bash status_web.sh		
		
※ 起動後、logs/ngrok.log に記載された公開URLへアクセスしてください。		
6. ディレクトリ構成		
Plaintext		
project/		
├── app/services/       # 各推論・処理エンジンの実装		
├── config/             # 空間定義(space_taxonomy.yaml)等の設定		
├── data/		
│   ├── uploads/        # 入力音声配置		
│   ├── segments/       # 分割済み音声（中間データ）		
│   └── results/        # 最終出力（JSON、画像）		
├── site/               # Webフロントエンド・サーバー管理		
│   ├── index.html      # ダッシュボードUI		
│   ├── start_web.sh    # Slurm起動スクリプト		
│   └── runtime/        # PID、ジョブID等の動的ファイル		
├── pipeline.py         # メインエントリーポイント		
└── requirements.txt    # 依存ライブラリ		
7. 制約・注意事項		
		
計算資源: VRAM 24GB 前後を想定しています。推論ごとにメモリ解放を行っていますが、モデルのロード順や競合にご注意ください。		
		
ネットワーク: 初回実行時は Hugging Face から合計数十GBのモデルデータをダウンロードします。		
		
Slurm運用: Webサーバーをジョブとして動かす場合、タイムアウト（デフォルト12時間）にご注意ください。不要になったら bash stop_web.sh でジョブを停止してください。		
8. ライセンス		
		
本プロジェクトのソースコードは MIT License です。		
		
※ 使用している各AIモデル（Llama-3.1-Swallow, Animagine XL等）については、それぞれのライセンス規定に従ってください。		
		
