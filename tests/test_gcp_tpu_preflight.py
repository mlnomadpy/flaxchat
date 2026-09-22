from scripts.gcp_tpu_preflight import effective_zone_limit


def test_zone_zero_override_takes_precedence_over_default():
    inventory = [{'metric': 'test', 'consumerQuotaLimits': [
        {'unit': '1/{project}/{zone}', 'quotaBuckets': [
            {'defaultLimit': '1536', 'effectiveLimit': '1536'},
            {'dimensions': {'zone': 'blocked-zone'}}]}]}]
    assert effective_zone_limit(inventory, 'test', 'blocked-zone') == 0
    assert effective_zone_limit(inventory, 'test', 'other-zone') == 1536
    assert effective_zone_limit(inventory, 'unknown', 'other-zone') is None


def test_regional_unlimited_bucket_is_not_zonal_quota():
    inventory = [{'metric': 'test', 'consumerQuotaLimits': [
        {'unit': '1/{project}/{region}', 'quotaBuckets': [{'effectiveLimit': '-1'}]}]}]
    assert effective_zone_limit(inventory, 'test', 'zone') is None
