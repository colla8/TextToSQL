# Text-to-SQL API

سرویس پرسش‌وپاسخ فارسی روی پایگاه‌داده فروشگاه: کاربر سؤال زبان طبیعی می‌پرسد،
سیستم schema مرتبط را با RAG پیدا می‌کند، یک LLM محلی (Ollama) کوئری SQL
تولید می‌کند، کوئری روی PostgreSQL اجرا می‌شود و نتیجه به فارسی خلاصه می‌شود.
یک ربات تلگرام هم روی همین API سوار است.

## معماری

```
Telegram user → api/bot/chatBot.py → POST /question → api/main.py
                                                          │
                                     ┌────────────────────┼─────────────────────┐
                                     ▼                    ▼                     ▼
                          app/pipeline.py         api/database.py (SQLite)   Postgres
                       (retrieval → SQL → run → answer)   وضعیت هر سؤال      داده‌ی فروشگاه
```

- **`app/`** — منطق اصلی pipeline (بدون تغییر رفتار، فقط باگ import و connection
  pooling اصلاح شد). دست نخورده باقی مانده جز دو مورد رفع باگ ذکرشده در پایین.
- **`api/`** — لایه‌ی سرویس‌دهی: FastAPI app، لایه‌ی پایگاه‌داده وضعیت
  درخواست‌ها، و ربات تلگرام.

## چه چیزی عوض شد و چرا

1. **باگ import در `app/llm/`** — `sql_generator.py` و `answer_generator.py`
   از ماژول `app.llm.prompts` (جمع) import می‌کردند در حالی که فایل واقعی
   `prompt.py` (مفرد) بود؛ یعنی پایپ‌لاین اصلاً import نمی‌شد. اصلاح شد.
2. **`api/main.py` واقعاً به pipeline وصل شد** — نسخه‌ی قبلی فقط یک stub بود
   و همیشه جواب `"سؤال دریافت شد: ..."` برمی‌گرداند بدون هیچ پردازشی.
3. **جلوگیری از تداخل کاربران هم‌زمان** —
   - `pipeline.ask()` کاملاً synchronous است (تماس شبکه‌ای با Ollama،
     Postgres، FAISS). اجرای مستقیمش داخل endpoint یعنی کل event loop
     برای همه‌ی کاربران دیگر هم قفل می‌شود. حالا با `asyncio.to_thread` در
     یک ترد جدا اجرا می‌شود و timeout هم دارد.
   - اتصال به Postgres قبلاً برای هر سؤال یک connection تازه باز می‌کرد
     (و در مسیرهای خطا هم بسته نمی‌شد). حالا از یک connection pool مشترک
     (`ThreadedConnectionPool`) استفاده می‌شود.
   - یک پایگاه‌داده‌ی SQLite سبک (`api/database.py`) اضافه شد که هر سؤال
     را با وضعیت pending/done/error ثبت می‌کند. اگر کاربری سؤال قبلی‌اش
     هنوز در حال پردازش باشد، درخواست جدیدش با کد ۴۰۹ رد می‌شود به‌جای
     این‌که هم‌زمان دو تا pipeline برایش اجرا شود.
   - مدل embedding و ایندکس FAISS فقط یک‌بار موقع بالا آمدن سرویس لود
     می‌شوند (نه هر درخواست).
4. **پیکربندی از `.env`** — قبلاً مقادیری مثل رمز عبور Postgres و مسیر
   مدل embedding داخل کد hardcode شده بودند (و مسیر `/home/zeinab/...` هم
   روی سرور دیگری اصلاً وجود ندارد). حالا همه از طریق `.env` تنظیم می‌شوند
   (`app/core/config.py` با `pydantic-settings`).
5. **باگ در `db_setup/load_data.py`** — مسیر `schema.sql` و مسیر CSVها
   اشتباه ساخته می‌شدند (`db_setup/db_setup/schema.sql` و
   `db_setup/dataset/csv`) و اسکریپت هرگز کار نمی‌کرد. اصلاح شد تا از
   `data/csv-data/csv/` واقعی بخواند.
6. **health check، لاگ‌گیری، مدیریت خطای یکدست، و endpoint تاریخچه** اضافه شد.

> ⚠️ **نکته‌ی امنیتی:** فایل `.env` آپلودشده شامل توکن واقعی ربات تلگرام
> بود. چون این فایل الان جای دیگری هم دیده شده، پیشنهاد می‌کنم توکن را از
> BotFather دوباره بسازید (revoke/regenerate) و مقدار جدید را فقط در
> `.env` (که هرگز commit نمی‌شود) نگه دارید.

## اجرای محلی (بدون Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # مقادیر واقعی را پر کنید (TOKEN, PG_*, ...)

# Postgres باید از قبل بالا باشد؛ سپس:
python db_setup/load_data.py --host localhost --user postgres --password postgres --db text2sql

# ساخت ایندکس embedding (اگر data/vector_db از قبل موجود نیست):
python scripts/build_metadata.py
python scripts/build_embeddings.py

# اجرای API:
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# در یک ترمینال دیگر، اجرای ربات تلگرام:
python -m api.bot.chatBot
```

Ollama هم باید جداگانه روی همان ماشین بالا باشد (`ollama serve`) و مدل
`qwen3:8b` (یا هرچه در `.env` تنظیم کرده‌اید) از قبل pull شده باشد.

## اجرای روی سرور با Docker

```bash
cp .env.example .env   # مقادیر واقعی را پر کنید
docker compose up -d --build
```

این سه سرویس را بالا می‌آورد: `db` (Postgres)، `api` (FastAPI)، `bot`
(ربات تلگرام). Ollama در compose نیامده چون معمولاً به GPU میزبان نیاز
دارد؛ روی خود سرور نصبش کنید (`ollama serve`) — کانتینر `api` با
`host.docker.internal` بهش وصل می‌شود (تنظیم `extra_hosts` در
`docker-compose.yml` همین را فراهم می‌کند).

بعد از بالا آمدن `db`، یک‌بار schema و داده را لود کنید:

```bash
docker compose exec api python db_setup/load_data.py --host db --user "$PG_USER" --password "$PG_PASSWORD" --db "$PG_DATABASE"
docker compose exec api python scripts/build_metadata.py
docker compose exec api python scripts/build_embeddings.py
```

برای قرار دادن پشت یک دامنه با HTTPS، یک reverse proxy (nginx / Caddy) جلوی
پورت ۸۰۰۰ بگذارید؛ Caddy این کار را با یک خط تنظیم می‌کند:

```
your-domain.com {
    reverse_proxy localhost:8000
}
```

## Endpointهای API

| Method | Path                    | توضیح                                              |
|--------|--------------------------|-----------------------------------------------------|
| GET    | `/health`                | وضعیت سرویس، اتصال Ollama و پایگاه‌داده             |
| POST   | `/question`              | `{telegram_id, question}` → `{answer, sql}`         |
| GET    | `/history/{telegram_id}` | تاریخچه‌ی سؤالات یک کاربر                           |

مستندات تعاملی (Swagger) به‌صورت خودکار روی `/docs` در دسترس است.

## تست سریع

```bash
curl -X POST http://localhost:8000/question \
  -H "Content-Type: application/json" \
  -d '{"telegram_id": 1, "question": "پرفروش‌ترین محصول چیست؟"}'
```
