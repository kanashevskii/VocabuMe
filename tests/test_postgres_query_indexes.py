import pytest
from django.db import connection


@pytest.mark.django_db
def test_vocabulary_hot_query_indexes_exist_on_postgres():
    if connection.vendor != "postgresql":
        pytest.skip("Index metadata is verified by the PostgreSQL CI job.")

    constraints = connection.introspection.get_constraints(
        connection.cursor(), "vocab_vocabularyitem"
    )

    for index_name in {
        "vocab_word_list_idx",
        "vocab_word_new_idx",
        "vocab_word_review_idx",
    }:
        assert index_name in constraints
        assert constraints[index_name]["index"] is True
