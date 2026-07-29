"""Telegram Stars payment handlers.

This module owns the Telegram transport boundary only. Payment validation and
subscription state transitions remain in the application service layer.
"""

from __future__ import annotations

import logging

from asgiref.sync import sync_to_async
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice, Update
from telegram.ext import ContextTypes

from vocab.integrations.telegram.messaging import safe_answer, safe_reply
from vocab.integrations.telegram.users import get_or_create_user
from vocab.monetization import (
    TELEGRAM_STARS_CURRENCY,
    get_telegram_stars_prices_for_user,
)
from vocab.services import (
    activate_subscription_for_successful_payment as activate_subscription_for_successful_payment_service,
    create_bot_payment_attempt,
    get_subscription_plans,
    validate_telegram_pre_checkout,
)


@sync_to_async
def get_paid_subscription_plans():
    return list(get_subscription_plans())


@sync_to_async
def activate_subscription_for_successful_payment(
    invoice_payload: str,
    telegram_payment_charge_id: str,
    provider_payment_charge_id: str,
    amount_minor: int,
    currency: str,
):
    return activate_subscription_for_successful_payment_service(
        invoice_payload=invoice_payload,
        telegram_payment_charge_id=telegram_payment_charge_id,
        provider_payment_charge_id=provider_payment_charge_id,
        amount_minor=amount_minor,
        currency=currency,
    )


async def payment_support(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(
        update,
        "Поддержка по оплатам VocabuMe: напиши сюда командой /paysupport и опиши проблему с Premium. "
        "Покупки обрабатываются внутри Telegram Stars; Telegram support не сможет решить вопросы по доступу в VocabuMe.",
    )


async def terms(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(
        update,
        "VocabuMe Premium дает цифровой доступ к расширенным сценариям, лимитам и AI-функциям в Mini App. "
        "Оплата проходит в Telegram Stars. По вопросам доступа, ошибок оплаты или возврата напиши /paysupport.",
    )


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    plans = await get_paid_subscription_plans()
    if not plans:
        await safe_reply(update, "Планы подписки пока недоступны.")
        return

    buttons = []
    chat_id = update.effective_chat.id if update.effective_chat else None
    stars_prices = get_telegram_stars_prices_for_user(chat_id)
    for plan in plans:
        period_label = "месяц" if plan.billing_period == "monthly" else "год"
        stars_amount = stars_prices.get(plan.billing_period)
        if stars_amount is None:
            continue
        buttons.append(
            [
                InlineKeyboardButton(
                    f"{stars_amount} Stars / {period_label}",
                    callback_data=f"subscribe:{plan.billing_period}",
                )
            ]
        )
    if not buttons:
        await safe_reply(update, "Планы подписки пока недоступны.")
        return
    await safe_reply(
        update,
        "💎 Premium открывает все relocation-сценарии и убирает лимиты.\nВыбери вариант подписки:",
        reply_markup=InlineKeyboardMarkup(buttons),
    )


async def start_subscription_checkout(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    query = update.callback_query
    await safe_answer(query)

    billing_period = query.data.split(":", 1)[1]
    plans = await get_paid_subscription_plans()
    plan = next((item for item in plans if item.billing_period == billing_period), None)
    if plan is None:
        await query.edit_message_text("План подписки недоступен.")
        return

    user, _ = await get_or_create_user(query.from_user.id, query.from_user.username)
    payment_attempt = await sync_to_async(create_bot_payment_attempt)(
        user, plan_code="premium", billing_period=billing_period
    )
    prices = [LabeledPrice(label=plan.name, amount=payment_attempt["amount_minor"])]
    await context.bot.send_invoice(
        chat_id=query.message.chat_id,
        title=f"{plan.name} for VocabuMe",
        description="Premium для безлимитного добавления и всех relocation-сценариев.",
        payload=payment_attempt["invoice_payload"],
        provider_token="",
        currency=TELEGRAM_STARS_CURRENCY,
        prices=prices,
    )
    await query.edit_message_text("Счёт отправлен выше. Заверши оплату в Telegram.")


async def handle_pre_checkout_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.pre_checkout_query
    if query is None:
        return
    valid, error_message = await sync_to_async(validate_telegram_pre_checkout)(
        invoice_payload=query.invoice_payload,
        amount_minor=query.total_amount,
        currency=query.currency,
    )
    await query.answer(ok=valid, error_message=error_message or None)


async def handle_successful_payment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    payment = update.message.successful_payment if update.message else None
    if payment is None:
        return
    try:
        subscription = await activate_subscription_for_successful_payment(
            invoice_payload=payment.invoice_payload,
            telegram_payment_charge_id=payment.telegram_payment_charge_id,
            provider_payment_charge_id=payment.provider_payment_charge_id,
            amount_minor=payment.total_amount,
            currency=payment.currency,
        )
    except Exception:
        logging.exception("Failed to activate subscription after successful payment")
        await safe_reply(
            update,
            "Платёж получен, но активация Premium задержалась. Мы уже разбираемся.",
        )
        return

    period_label = "месяц" if subscription.plan.billing_period == "monthly" else "год"
    await safe_reply(
        update,
        f"✅ Premium активирован на {period_label}.\nДоступ открыт до {subscription.expires_at:%d.%m.%Y}.",
    )
