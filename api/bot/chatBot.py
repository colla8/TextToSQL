import asyncio
import logging

import httpx

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage

from app.core.config import setting

if not setting.TOKEN:
    raise ValueError("TOKEN تنظیم نشده است (در .env مقدار TOKEN را قرار دهید).")


bot = Bot(token=setting.TOKEN)
dp = Dispatcher(storage=MemoryStorage())

logger = logging.getLogger(__name__)

FASTAPI_URL = setting.FASTAPI_URL


class QuestionState(StatesGroup):
    waiting_for_question = State()


@dp.message(CommandStart())
async def start(message: Message, state: FSMContext):
    await state.clear()

    await message.answer("hii🧚‍♀️")


@dp.message(Command("help"))
async def help_command(message: Message, state: FSMContext):
    await state.clear()

    await message.answer(
        "/question: برای ارسال سؤال استفاده کنید.\n"
        "/help: نمایش راهنما\n"
        "/start: شروع مجدد"
    )


@dp.message(Command("question"))
async def asking(message: Message, state: FSMContext):

    await state.set_state(QuestionState.waiting_for_question)

    await message.answer("سؤال خود را ارسال کنید:")


async def _post_question(telegram_id: int, question: str) -> httpx.Response:
    # timeout=None: a question can legitimately take several minutes on
    # this hardware. Duration alone is never a reason to give up -- see
    # _watch_server_alive for how an actual crash is detected instead.
    async with httpx.AsyncClient(timeout=None) as client:
        return await client.post(
            f"{FASTAPI_URL}/question",
            json={"telegram_id": telegram_id, "question": question},
        )


async def _watch_server_alive(request_task: asyncio.Task, message: Message) -> None:
    """Runs alongside the /question request, without limiting how long
    processing may take. Every CHECK_INTERVAL seconds it pings /health
    with its own short timeout. /health answers instantly even while a
    question is being processed (it's a separate async endpoint, and
    pipeline.ask() runs in a worker thread -- see api/main.py), so
    repeated failures mean the FastAPI process itself is actually gone
    (crashed / killed / OOM), not just busy. Only then do we cancel the
    request and tell the user, instead of leaving them waiting on a
    server that no longer exists.
    """
    CHECK_INTERVAL_SECONDS = 8
    MAX_CONSECUTIVE_FAILURES = 3

    consecutive_failures = 0
    while not request_task.done():
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)
        if request_task.done():
            return

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{FASTAPI_URL}/health")
            consecutive_failures = (
                0 if resp.status_code == 200 else consecutive_failures + 1
            )
        except httpx.HTTPError:
            consecutive_failures += 1

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            request_task.cancel()
            await message.answer(
                "سرویس در حین پردازش سؤال شما از کار افتاد (کرش کرد). "
                "لطفاً دوباره تلاش کنید."
            )
            return


@dp.message(QuestionState.waiting_for_question, F.text)
async def receive_question(message: Message, state: FSMContext):

    question = message.text.strip()

    if not question:
        await message.answer("لطفاً سؤال خود را ارسال کنید.")
        return

    await message.answer("سؤال شما دریافت شد.\n" "در حال پردازش...")

    request_task = asyncio.create_task(_post_question(message.from_user.id, question))
    watchdog_task = asyncio.create_task(_watch_server_alive(request_task, message))

    try:

        response = await request_task

        if response.status_code == 409:
            await message.answer("سؤال قبلی شما هنوز در حال پردازش است. کمی صبر کنید.")
            return

        response.raise_for_status()

        data = response.json()

        answer = data.get("answer")

        if not answer:
            await message.answer("پاسخی از سرور دریافت نشد.")
            return

        await message.answer(answer)

    except asyncio.CancelledError:

        # _watch_server_alive already cancelled this task and told the
        # user about the crash -- nothing more to do here.
        pass

    except httpx.HTTPError as e:

        logger.error(f"FastAPI error: {e}")

        await message.answer("خطایی در ارتباط با سرور رخ داد.")

    except Exception as e:

        logger.exception(e)

        await message.answer("یک خطای غیرمنتظره رخ داد.")

    finally:

        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass

        # بعد از دریافت سؤال، از حالت انتظار خارج می‌شویم
        await state.clear()


async def main():

    logging.basicConfig(level=logging.INFO)

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
