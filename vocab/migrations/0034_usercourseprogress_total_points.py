from django.db import migrations, models


def backfill_total_points(apps, schema_editor):
    TelegramUser = apps.get_model("vocab", "TelegramUser")
    UserCourseProgress = apps.get_model("vocab", "UserCourseProgress")

    for progress in UserCourseProgress.objects.all().iterator():
        progress.total_points = (
            (progress.practice_correct or 0)
            + (progress.listening_correct or 0)
            + (progress.speaking_correct or 0)
            + (progress.review_correct or 0)
            + (progress.irregular_correct or 0)
        )
        progress.save(update_fields=["total_points"])

    users_without_progress = TelegramUser.objects.exclude(
        id__in=UserCourseProgress.objects.values_list("user_id", flat=True)
    )
    for user in users_without_progress.iterator():
        course_code = user.active_studied_language or "en"
        UserCourseProgress.objects.create(
            user_id=user.id,
            course_code=course_code,
            total_study_days=user.total_study_days or 0,
            consecutive_days=user.consecutive_days or 0,
            last_study_date=user.last_study_date,
            irregular_correct=user.irregular_correct or 0,
            practice_correct=user.practice_correct or 0,
            listening_correct=user.listening_correct or 0,
            speaking_correct=user.speaking_correct or 0,
            review_correct=user.review_correct or 0,
            total_points=(
                (user.practice_correct or 0)
                + (user.listening_correct or 0)
                + (user.speaking_correct or 0)
                + (user.review_correct or 0)
                + (user.irregular_correct or 0)
            ),
        )


class Migration(migrations.Migration):

    dependencies = [
        ("vocab", "0033_telegramuser_custom_avatar_url"),
    ]

    operations = [
        migrations.AddField(
            model_name="usercourseprogress",
            name="total_points",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.RunPython(backfill_total_points, migrations.RunPython.noop),
    ]
