# Benchmark harness

اضافه‌شده به `Text-to-SQL/`، بدون تغییر در `app/` یا `api/`. کاری که می‌کند:

هر مرحله‌ی `pipeline.ask()` را جدا زمان می‌گیرد (`retrieval`, `sql_generation`,
`sql_execution`, `answer_generation`)، و هم‌زمان با یک ترد جدا CPU%/RSS پردازش
`ollama` و `postgres` و (اگر GPU باشد) util%/VRAM را با فاصله‌ی ثابت
نمونه‌برداری می‌کند. برای هر مرحله، بازه‌ی زمانی همان مرحله از نمونه‌ها جدا
می‌شود. علاوه بر این، پاسخ خام Ollama (`load_duration`, `prompt_eval_duration`,
`eval_duration`, `eval_count`) هم برای هر تماس LLM ذخیره می‌شود — این یعنی
tokens/sec واقعی و سهم reload مدل را بدون هیچ نمونه‌برداری اضافه‌ای داری.

## قبل از اجرا

باید همان چیزهایی که `api/main.py` لازم دارد در دسترس باشند:
- Postgres روشن و پرشده (`db_setup/load_data.py` قبلاً اجرا شده)
- `ollama serve` روشن و مدل `qwen3:8b` (یا هرچه در `.env` است) pull‌شده
- embedding index ساخته‌شده (`data/vector_db` موجود)
- همان venv پروژه (`pip install psutil` اگر نیست؛ برای نمودار
  `pip install matplotlib` هم اختیاری است)

نمونه‌بردار به‌صورت پیش‌فرض دنبال `ollama`, `llama-server`, `postgres` می‌گردد.
روی نصب snap (مثل نصب خودت) `ollama serve` فقط یک wrapper سبک است -- کار واقعی
(و مصرف CPU/RAM/GPU) در یک پردازش فرزند به نام `llama-server` انجام می‌شود.
اگر خروجی `ps aux | grep -E "ollama|llama-server"` روی سیستمت اسم دیگری نشان
داد، آن را در `ResourceSampler(watch_process_names=(...))` در
`run_benchmark.py` جایگزین کن.

نکته‌ی مجوز: اگر `ollama`/`llama-server` با کاربر دیگری (مثلاً `root` زیر
snap) اجرا می‌شوند و تو با کاربر عادی بنچمارک را اجرا می‌کنی، ممکن است
psutil به آن پردازش دسترسی نداشته باشد. در این حالت هنگام `start()` یک هشدار
چاپ می‌شود («دسترسی به پردازش ... نداری») -- در آن صورت یا با `sudo` اجرا کن،
یا بپذیر که ستون‌های CPU/RAM آن پردازش در گزارش خالی می‌مانند (زمان‌بندی هر
مرحله و آمار GPU از `nvidia-smi` تحت تأثیر قرار نمی‌گیرند، چون آن‌ها متکی به
دسترسی به آن پردازش خاص نیستند).

## اجرا

```bash
cd Text-to-SQL
python -m benchmarks.run_benchmark --repeats 5
python -m benchmarks.report benchmarks/results/run_YYYYMMDD_HHMMSS.jsonl
```

`--repeats 5` یعنی هر سؤال در `questions.py` پنج بار تکرار می‌شود. کمتر از این
آماره‌ی median/p95 بی‌معنی می‌شود چون inference نویز واقعی دارد. اگر فقط یک
دسته را می‌خواهی: `--only-category aggregation`.

نتیجه یک JSONL خام است (هر خط = یک اجرای یک سؤال، شامل تک‌تک نمونه‌های منابع
هر مرحله) — برای بازبینی دستی یا تحلیل جدا هم قابل استفاده است، نه فقط برای
`report.py`.

## خواندن گزارش

- **زمان هر مرحله (median/p95)**: کدام مرحله واقعاً وقت می‌برد. تقریباً حتم
  دارم `sql_generation` غالب می‌شود چون `think=True` است و روی همین سخت‌افزار
  (CPU-heavy طبق کامنت‌های خودت در `config.py`) هر توکن گران است.
- **tokens/sec میانگین**: اگر پایین است (مثلاً زیر ۱۰-۱۵ روی CPU)، مشکل خام
  سرعت inference است — راه‌حل معمولاً `NUM_GPU` بالاتر (اگر VRAM جا دارد)،
  quantization سبک‌تر مدل، یا کوتاه کردن prompt/`NUM_CTX` است، نه معماری
  pipeline.
- **calls با load_duration>0 بیشتر از ۱ در کل اجرا**: یعنی `keep_alive` کار
  نمی‌کند و مدل مدام reload می‌شود — این خودش می‌تواند چند ثانیه به هر سؤال
  اضافه کند و قبل از هر بهینه‌سازی دیگری باید رفعش کرد.
- **repair_attempts** (در `meta` مرحله‌ی `sql_generation`، در فایل JSONL خام):
  اگر مرتب بالای صفر است، مشکل کیفیت retrieval یا prompt است، نه سرعت — یک
  تماس دوم LLM یعنی دو برابر هزینه‌ی آن مرحله برای همان سؤال.
- **peak ollama RSS**: اگر نزدیک سقف RAM سیستم است، این توضیح می‌دهد چرا OS
  ممکن است swap کند یا OOM بدهد زیر بار هم‌زمان.

## محدودیت‌های شناخته‌شده

- نمونه‌برداری منابع فاصله‌محور است (پیش‌فرض ۱۰۰ms). مرحله‌ای سریع‌تر از این
  (مثلاً `sql_execution` روی این دیتاست کوچک، یا خود `sql_validator`) اصلاً
  نمونه‌ای نمی‌گیرد -- گزارش این را با `n_samples: 0` / «نمونه‌ای ثبت نشد» نشان
  می‌دهد، نه با صفر جعلی.
- فقط GPU اول را می‌خواند (`nvidia-smi` خط اول). اگر چند GPU داری، خودت
  `_sample_gpu` در `resource_monitor.py` را برای همه‌شان گسترش بده.
- این هارنس، اجراهای متوالی و تک‌کاربره است — رفتار هم‌زمانی چند کاربر
  (که `README.md` اصلی پروژه به آن اشاره کرده: connection pool،
  `asyncio.to_thread`، قفل ۴۰۹ روی سؤال درحال پردازش) را benchmark نمی‌کند.
  اگر آن هم مهم است، باید یک هارنس جدا با چند کلاینت همزمان علیه خود API
  (نه مستقیم علیه pipeline) نوشت — چیز متفاوتی است، چون آنجا queueing و
  connection pool exhaustion سنجیده می‌شود، نه سرعت خام یک درخواست.
