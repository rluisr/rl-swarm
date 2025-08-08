# RL Swarm プロジェクト構造

## 主要コンポーネント
- **GenRL**: `gensyn-genrl==0.1.4` パッケージとして外部からインストール
- **トレーナー**: `rgym_exp/src/trainer.py` の `GRPOTrainerModule` クラス
  - 親クラス: `genrl.trainer.grpo_trainer.GRPOLanguageTrainerModule`
- **モデル設定**: `rgym_exp/config/rg-swarm.yaml`
  - モデルロード: `transformers.AutoModelForCausalLM.from_pretrained`
  - GPU選択: `omega_gpu_resolver.py` でVRAMベースでモデル選択

## 設定ファイル
- メイン設定: `rgym_exp/config/rg-swarm.yaml`
- データセット設定: `rgym_exp/src/datasets.yaml`

## CUDA関連
- `trainer.py:72`: `input_ids.to(self.model.device)`でモデルのデバイスを使用
- `omega_gpu_resolver.py`: GPU VRAMチェック機能

## 起動方法
- シェルスクリプト: `run_rl_swarm.sh`
- Docker Compose: `docker-compose.yaml`