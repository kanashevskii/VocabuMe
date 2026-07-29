from vocab import services
from vocab.application import entitlements


def test_services_keeps_entitlement_compatibility_facade():
    assert services.EntitlementError is entitlements.EntitlementError
    assert (
        services.reserve_new_items_for_today is entitlements.reserve_new_items_for_today
    )
    assert services.ensure_pack_is_accessible is entitlements.ensure_pack_is_accessible
    assert services.user_has_premium is entitlements.user_has_premium
