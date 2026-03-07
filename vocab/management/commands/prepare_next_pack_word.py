from django.core.management.base import BaseCommand

from vocab.services import prepare_next_pack_word


class Command(BaseCommand):
    help = "Prepare one next pending word from configured word packs."

    def handle(self, *args, **options):
        prepared = prepare_next_pack_word()
        if prepared is None:
            self.stdout.write("No pending pack words.")
            return
        self.stdout.write(
            self.style.SUCCESS(
                f"Prepared {prepared.pack_id}/{prepared.level_id}: {prepared.word}"
            )
        )
