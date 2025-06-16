import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

from django.db import IntegrityError
from vocab.models import VocabularyItem
from vocab.utils import clean_word

for item in VocabularyItem.objects.all().order_by("id"):
    new_word = clean_word(item.word)
    changed = False
    if item.word != new_word:
        item.word = new_word
        changed = True
    if item.normalized_word != new_word:
        item.normalized_word = new_word
        changed = True
    if changed:
        try:
            item.save()
            print(f"Updated {item.id} -> {new_word}")
        except IntegrityError:
            dup = VocabularyItem.objects.filter(user=item.user, normalized_word=new_word).exclude(id=item.id).first()
            if dup:
                print(f"Deleting duplicate {item.id} ({new_word})")
                item.delete()
            else:
                print(f"Failed to update {item.id}")
