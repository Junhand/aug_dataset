import os
import asyncio
import json
import requests
import time

from dotenv import load_dotenv

load_dotenv()


class LocalVLLMClient:
    """
    vLLM/OpenAI互換APIサーバ用のクライアント
    - 非同期API: generate_text() は await 可能
    - 同期HTTP（requests）を asyncio.to_thread で回すので追加依存なし
    - 環境変数:
        OPENAI_BASE_URL (例: http://localhost:8049/v1)
        OPENAI_API_KEY  (必要なら)
    """

    def __init__(
        self,
        model_path: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout_connect: float = 20.0,
        timeout_read: float = 3000.0,
        max_retries: int = 3,
        repetition_penalty: float | None = 1.1,
        enable_thinking: bool | None = True,
    ):
        # 既存シグネチャ互換のため model_path をそのままモデルIDとして使う
        self.model = model_path
        self.temperature = temperature
        self.default_max_tokens = max_tokens
        self.base_url = os.getenv("MODEL_BASE_URL", "http://localhost:8000/v1")
        self.api_key = os.getenv("MODEL_API_KEY", "")
        self.timeout_connect = timeout_connect
        self.timeout_read = timeout_read
        self.max_retries = max_retries
        self.repetition_penalty = repetition_penalty
        self.enable_thinking = enable_thinking

        # /chat/completions エンドポイント
        self._chat_url = f"{self.base_url}/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {self.api_key or 'EMPTY'}",
            "Content-Type": "application/json",
        }

    async def generate_text(self, prompt: str, max_tokens: int | None = None) -> str:
        """
        単一プロンプトを user メッセージとして送信し、アシスタントの content を文字列で返す
        - 返り値は生テキスト（JSONは上位で抽出）
        """
        payload = self._build_payload(prompt, max_tokens)

        # 同期HTTPをスレッドで回してawait
        resp_json = await asyncio.to_thread(
            self._post_with_retries, self._chat_url, self._headers, payload
        )
        try:
            return resp_json["choices"][0]["message"]["content"]
        except Exception:
            # 互換性のため多少ゆるくフォールバック
            return json.dumps(resp_json, ensure_ascii=False)

    # ---------------- 内部実装 ----------------

    def _build_payload(self, prompt: str, max_tokens: int | None):
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": int(max_tokens or self.default_max_tokens),
        }
        # ある実装では repetition_penalty を受け付ける（任意）
        if self.repetition_penalty is not None:
            payload["repetition_penalty"] = float(self.repetition_penalty)

        # 思考展開テンプレ対応がある環境のみ（任意）
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {
                "enable_thinking": bool(self.enable_thinking)
            }

        return payload

    def _post_with_retries(self, url: str, headers: dict, json_payload: dict) -> dict:
        """
        指数バックオフ＋ジッタで /chat/completions を叩く
        - 429/5xx はリトライ
        - タイムアウトは (接続, 読み取り) で設定
        """
        last_err: Exception | None = None
        attempts = self.max_retries + 1  # 例: max_retries=3 -> 4回試行
        for attempt in range(1, attempts + 1):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=json_payload,
                    timeout=(self.timeout_connect, self.timeout_read),
                )
                if resp.status_code == 200:
                    return resp.json()

                # リトライ対象のステータス
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < attempts:
                    delay = min(2 ** (attempt - 1), 30)
                    time.sleep(delay)
                    continue

                # その他は即エラー
                raise RuntimeError(
                    f"vLLM API error {resp.status_code}: {resp.text[:500]}"
                )

            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = e
                if attempt < attempts:
                    delay = min(2 ** (attempt - 1), 30)
                    time.sleep(delay)
                    continue
                break

        raise RuntimeError(f"vLLM API request failed after retries: {last_err}")
