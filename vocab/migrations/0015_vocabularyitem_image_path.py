from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('vocab', '0014_telegramuser_reminder_timezone'),
    ]

    operations = [
        migrations.AddField(
            model_name='vocabularyitem',
            name='image_path',
            field=models.CharField(blank=True, default='', max_length=500),
        ),
    ]
