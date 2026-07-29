from vocab import services
from vocab.application import billing


def test_services_keeps_billing_compatibility_facade():
    assert services.create_checkout_session is billing.create_checkout_session
    assert (
        services.activate_subscription_for_successful_payment
        is billing.activate_subscription_for_successful_payment
    )
    assert (
        services.validate_telegram_pre_checkout
        is billing.validate_telegram_pre_checkout
    )
    assert services.get_billing_payload is billing.get_billing_payload
