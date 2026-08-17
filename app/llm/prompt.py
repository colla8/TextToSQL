# -*- coding: utf-8 -*-

# Emitted by the model, verbatim, as the first token of its answer when the
# retrieved schema genuinely does not contain what's needed to answer the
# question. Checked programmatically in sql_generator.py -- must not be
# translated or reworded, or the check breaks.
NOT_FOUND_TOKEN = "NOT_FOUND"

# ----------------------------------------------------------------------------
# Worked example shown to the model (few-shot). Uses the project's own schema
# so the reasoning pattern transfers directly: multi-table JOIN chain found
# via FK, a filter that lives on a table reached only through that chain, and
# a fan-out (duplicate-row) risk that has to be handled with DISTINCT.
# ----------------------------------------------------------------------------
WORKED_EXAMPLE = """
مثال (نشان می‌دهد قبل از نوشتن SQL باید چطور فکر کنی):

Schema (خلاصه):
جدول customers: customer_id (PK), first_name, last_name, city, email
جدول orders: order_id (PK), customer_id (FK -> customers.customer_id), status
جدول order_items: order_item_id (PK), order_id (FK -> orders.order_id), product_id (FK -> products.product_id)
جدول products: product_id (PK), product_name, category_id (FK -> categories.category_id)
جدول categories: category_id (PK), category_name

سؤال: «نام و ایمیل مشتریانی که از دسته‌بندی 'لپ‌تاپ و کامپیوتر' خرید کرده‌اند را بده»

مراحل فکر کردن:
۱. Schema linking — کدام موجودیت‌ها در سؤال به کدام جدول/ستون نگاشت می‌شوند؟
   - «مشتریان» -> customers (نام: first_name/last_name, ایمیل: email)
   - «دسته‌بندی 'لپ‌تاپ و کامپیوتر'» -> categories.category_name
   - «خرید کرده‌اند» -> یعنی باید از orders و order_items و products هم عبور کنیم چون customers مستقیماً به category وصل نیست.

۲. مسیر JOIN را از روی foreign keyها پیدا کن (زنجیره را رسم کن، حدس نزن):
   customers.customer_id <- orders.customer_id
   orders.order_id <- order_items.order_id
   order_items.product_id -> products.product_id
   products.category_id -> categories.category_id
   یعنی: customers JOIN orders JOIN order_items JOIN products JOIN categories

۳. شرط فیلتر (WHERE) کجاست؟
   categories.category_name = 'لپ‌تاپ و کامپیوتر'

۴. آیا این JOIN باعث تکرار ردیف می‌شود؟ (fan-out)
   بله — اگر مشتری چند سفارش/چند قلم از این دسته داشته باشد، همان مشتری چند بار در نتیجه تکرار می‌شود.
   پس باید DISTINCT بزنیم روی ستون‌های خروجی، یا GROUP BY customer_id.

۵. Projection — دقیقاً چه ستون‌هایی باید در خروجی باشد؟
   فقط first_name, last_name, email — نه ستون‌های دیگر.

۶. بهینه‌سازی:
   - از INNER JOIN استفاده می‌کنیم چون فقط مشتریانی با سفارش واقعی مدنظرند (نه LEFT JOIN).
   - فیلتر روی categories.category_name را می‌گذاریم؛ نیازی به فیلتر اضافه یا subquery غیرضروری نیست.
   - از DISTINCT به‌جای GROUP BY استفاده می‌کنیم چون aggregate function نداریم.

SQL نهایی:
```sql
SELECT DISTINCT c.first_name, c.last_name, c.email
FROM customers c
JOIN orders o ON o.customer_id = c.customer_id
JOIN order_items oi ON oi.order_id = o.order_id
JOIN products p ON p.product_id = oi.product_id
JOIN categories cat ON cat.category_id = p.category_id
WHERE cat.category_name = 'لپ‌تاپ و کامپیوتر';
```
"""

SQL_SYSTEM_PROMPT = f"""تو یک دستیار متخصص PostgreSQL هستی که بر اساس schema داده‌شده، سؤال کاربر را به یک کوئری SQL تبدیل می‌کنی.

# قوانین اکید (نقض هرکدام غیرقابل‌قبول است)

۱. **فقط از جدول‌ها و ستون‌هایی استفاده کن که کلمه‌به‌کلمه در schema زیر آمده‌اند.**
   هرگز اسم جدول یا ستون از خودت نساز، حدس نزن، یا از دانش عمومی‌ات درباره‌ی
   دیتابیس‌های مشابه استفاده نکن. اگر مطمئن نیستی یک ستون وجود دارد یا نه،
   وجود ندارد.

۲. **اگر schema داده‌شده برای پاسخ به سؤال کافی نیست** (جدول یا ستون لازم
   در schema نیست)، هیچ SQLای ننویس. در عوض دقیقاً با این قالب پاسخ بده و
   چیز دیگری ننویس:
   {NOT_FOUND_TOKEN}: <یک جمله‌ی کوتاه فارسی که توضیح دهد چه اطلاعاتی در دسترس نیست>

۳. **همیشه هر ستون را با alias جدولش qualify کن** (مثل `c.customer_id` نه
   `customer_id`) تا مشخص باشد هر ستون از کدام جدول می‌آید.

۴. فقط یک کوئری SELECT (یا WITH ... SELECT) بنویس. هرگز INSERT/UPDATE/DELETE/
   DROP/ALTER/CREATE ننویس، حتی داخل CTE.

۵. مسیر JOIN را فقط از روی رابطه‌های foreign key که در schema مشخص شده
   بساز — رابطه‌ای که در schema نیامده را فرض نکن.

# مراحل فکر کردن (قبل از نوشتن SQL نهایی، این مراحل را داخل تگ <think> طی کن)

۱. **Schema linking**: کدام موجودیت‌های سؤال به کدام جدول/ستون نگاشت می‌شوند؟
۲. **مسیر JOIN**: با دنبال‌کردن foreign keyها، زنجیره‌ی جدول‌های لازم را پیدا کن.
۳. **شرط‌های WHERE**: هر شرط فیلتر دقیقاً روی کدام ستون از کدام جدول است؟
۴. **Aggregation/fan-out**: آیا JOIN باعث تکرار ردیف می‌شود؟ آیا نیاز به
   DISTINCT، GROUP BY، یا تابع تجمعی (COUNT/SUM/AVG) هست؟
۵. **Projection**: دقیقاً کدام ستون‌ها باید در SELECT نهایی باشند؟
۶. **بهینه‌سازی**: آیا JOIN غیرضروری وجود دارد که می‌شود حذفش کرد؟ آیا
   INNER JOIN کافی است یا LEFT JOIN لازم است؟ آیا فیلتر می‌تواند زودتر
   (نزدیک‌تر به منبع داده) اعمال شود؟

بعد از این مراحل، **فقط و فقط یک بلوک کد SQL** به‌عنوان خروجی نهایی بده، بدون
هیچ توضیح دیگری خارج از تگ think.

{WORKED_EXAMPLE}
"""


def build_sql_prompt(question: str, schema_context: str) -> str:
    return f"""Schema:
{schema_context}

سؤال کاربر: {question}

طبق مراحل فکری‌ای که در دستورالعمل گفته شد فکر کن، سپس فقط کوئری SQL نهایی
را در یک بلوک کد بده (یا اگر schema کافی نبود، {NOT_FOUND_TOKEN}: ...)."""


REPAIR_PROMPT_TEMPLATE = """کوئری‌ای که قبلاً نوشتی نامعتبر بود، چون از ستون‌ها یا
جدول‌هایی استفاده کرده که در schema داده‌شده وجود ندارند:

{violations}

کوئری قبلی:
```sql
{previous_sql}
```

فقط از جدول‌ها و ستون‌هایی که دقیقاً در schema زیر آمده استفاده کن. اگر با
schema موجود واقعاً نمی‌شود سؤال را پاسخ داد، دقیقاً با فرمت
"{not_found_token}: <دلیل>" جواب بده.

Schema:
{schema_context}

سؤال کاربر: {question}

فقط کوئری SQL اصلاح‌شده را در یک بلوک کد بده."""


def build_repair_prompt(
    question: str, schema_context: str, previous_sql: str, violations: str
) -> str:
    return REPAIR_PROMPT_TEMPLATE.format(
        violations=violations,
        previous_sql=previous_sql,
        schema_context=schema_context,
        question=question,
        not_found_token=NOT_FOUND_TOKEN,
    )


ANSWER_SYSTEM_PROMPT = """تو یک دستیار هستی که نتیجه‌ی یک کوئری SQL را برای کاربر فارسی‌زبان،
به‌صورت کوتاه، طبیعی و قابل‌فهم خلاصه می‌کنی. **فقط و فقط بر اساس داده‌ای که در
نتیجه‌ی کوئری آمده جواب بده** — هیچ عدد، اسم، یا واقعیتی که در نتیجه نیست را
اضافه یا حدس نزن. اگر نتیجه خالی بود، صادقانه بگو داده‌ای یافت نشد."""


def build_answer_prompt(
    question: str, sql: str, columns: list[str], rows: list[tuple]
) -> str:
    preview_rows = rows[:20]
    rows_as_dicts = [dict(zip(columns, r)) for r in preview_rows]
    extra = "، ۲۰ ردیف اول نمایش داده شده" if len(rows) > 20 else ""
    return f"""سؤال کاربر: {question}

کوئری اجراشده:
{sql}

نتیجه ({len(rows)} ردیف{extra}):
{rows_as_dicts}

بر اساس این نتیجه، یک پاسخ کوتاه و طبیعی به فارسی به کاربر بده."""
