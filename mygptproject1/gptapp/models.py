from django.db import models


class Feedback(models.Model):
    """Stores user feedback on agent-generated responses."""

    RATING_CHOICES = [
        ('positive', 'Positive'),
        ('negative', 'Negative'),
    ]

    OUTPUT_TYPE_CHOICES = [
        ('CSV', 'CSV'),
        ('IMAGE', 'Image'),
        ('PDF', 'PDF'),
        ('TEXT', 'Text'),
    ]

    # What the user asked
    query_text = models.TextField(help_text="User's original prompt")
    output_type = models.CharField(max_length=10, choices=OUTPUT_TYPE_CHOICES)

    # What the agent produced (truncated summary for prompt injection)
    response_summary = models.TextField(
        help_text="Short summary or first ~500 chars of the agent response"
    )

    # User's judgement
    rating = models.CharField(max_length=10, choices=RATING_CHOICES)
    comment = models.TextField(blank=True, default='',
                               help_text="Optional free-text feedback from the user")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"[{self.rating}] {self.output_type} — {self.query_text[:60]}"
