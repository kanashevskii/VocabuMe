from django.db import migrations, models


DEFAULT_STUDIED_LANGUAGE = "en"


def seed_course_progress(apps, schema_editor):
    TelegramUser = apps.get_model("vocab", "TelegramUser")
    UserCourseProgress = apps.get_model("vocab", "UserCourseProgress")

    progress_rows = []
    for user in TelegramUser.objects.all().iterator():
        progress_rows.append(
            UserCourseProgress(
                user_id=user.id,
                course_code=DEFAULT_STUDIED_LANGUAGE,
                total_study_days=user.total_study_days,
                consecutive_days=user.consecutive_days,
                last_study_date=user.last_study_date,
                irregular_correct=user.irregular_correct,
                practice_correct=user.practice_correct,
                listening_correct=user.listening_correct,
                speaking_correct=user.speaking_correct,
                review_correct=user.review_correct,
            )
        )

    if progress_rows:
        UserCourseProgress.objects.bulk_create(progress_rows, ignore_conflicts=True)


class Migration(migrations.Migration):
    dependencies = [
        ("vocab", "0026_telegramuser_auth_provider_telegramuser_email_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="telegramuser",
            name="active_studied_language",
            field=models.CharField(
                choices=[("en", "English"), ("ka", "Georgian")],
                default="en",
                max_length=10,
            ),
        ),
        migrations.CreateModel(
            name="UserCourseProgress",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "course_code",
                    models.CharField(
                        choices=[("en", "English"), ("ka", "Georgian")],
                        default="en",
                        max_length=10,
                    ),
                ),
                ("total_study_days", models.PositiveIntegerField(default=0)),
                ("consecutive_days", models.PositiveIntegerField(default=0)),
                ("last_study_date", models.DateField(blank=True, null=True)),
                ("irregular_correct", models.PositiveIntegerField(default=0)),
                ("practice_correct", models.PositiveIntegerField(default=0)),
                ("listening_correct", models.PositiveIntegerField(default=0)),
                ("speaking_correct", models.PositiveIntegerField(default=0)),
                ("review_correct", models.PositiveIntegerField(default=0)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=models.deletion.CASCADE, to="vocab.telegramuser"
                    ),
                ),
            ],
            options={"unique_together": {("user", "course_code")}},
        ),
        migrations.AddField(
            model_name="vocabularyitem",
            name="course_code",
            field=models.CharField(
                choices=[("en", "English"), ("ka", "Georgian")],
                default="en",
                max_length=10,
            ),
        ),
        migrations.AddField(
            model_name="achievement",
            name="course_code",
            field=models.CharField(
                choices=[("en", "English"), ("ka", "Georgian")],
                default="en",
                max_length=10,
            ),
        ),
        migrations.AddField(
            model_name="irregularverbprogress",
            name="course_code",
            field=models.CharField(
                choices=[("en", "English"), ("ka", "Georgian")],
                default="en",
                max_length=10,
            ),
        ),
        migrations.AddField(
            model_name="addworddraft",
            name="course_code",
            field=models.CharField(
                choices=[("en", "English"), ("ka", "Georgian")],
                default="en",
                max_length=10,
            ),
        ),
        migrations.AddField(
            model_name="packpreparedword",
            name="course_code",
            field=models.CharField(
                choices=[("en", "English"), ("ka", "Georgian")],
                default="en",
                max_length=10,
            ),
        ),
        migrations.RemoveConstraint(
            model_name="vocabularyitem",
            name="unique_user_normalized_word",
        ),
        migrations.AddConstraint(
            model_name="vocabularyitem",
            constraint=models.UniqueConstraint(
                fields=("user", "course_code", "normalized_word"),
                name="unique_user_course_normalized_word",
            ),
        ),
        migrations.AlterUniqueTogether(
            name="achievement",
            unique_together={("user", "course_code", "code")},
        ),
        migrations.AlterUniqueTogether(
            name="irregularverbprogress",
            unique_together={("user", "course_code", "verb_base")},
        ),
        migrations.AlterUniqueTogether(
            name="packpreparedword",
            unique_together={("course_code", "pack_id", "level_id", "normalized_word")},
        ),
        migrations.AlterModelOptions(
            name="packpreparedword",
            options={"ordering": ["course_code", "pack_id", "level_id", "word"]},
        ),
        migrations.RunPython(seed_course_progress, migrations.RunPython.noop),
    ]
