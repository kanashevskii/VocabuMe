from django.core.management.base import BaseCommand

from vocab.services import prepare_all_word_packs


class Command(BaseCommand):
    help = "Prepare translations, examples, and images for all configured word packs."

    def handle(self, *args, **options):
        self.stdout.write("Preparing word packs...")
        prepare_all_word_packs()
        self.stdout.write(self.style.SUCCESS("Word packs prepared."))
