"""
Feedback API views — collect and serve user feedback on agent responses.
"""
import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .models import Feedback

logger = logging.getLogger(__name__)


@csrf_exempt
@require_http_methods(["POST"])
def submit_feedback(request):
    """
    POST /api/feedback/
    Body: {
        "query_text": "...",
        "output_type": "CSV|IMAGE|PDF|TEXT",
        "response_summary": "...",
        "rating": "positive|negative",
        "comment": "..."   (optional)
    }
    """
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError) as e:
        return JsonResponse({'error': f'Invalid JSON: {e}'}, status=400)

    query_text = (data.get('query_text') or '').strip()
    output_type = (data.get('output_type') or '').upper()
    response_summary = (data.get('response_summary') or '').strip()
    rating = (data.get('rating') or '').lower()
    comment = (data.get('comment') or '').strip()

    # Validate required fields
    if not query_text:
        return JsonResponse({'error': 'query_text is required'}, status=400)
    if output_type not in ('CSV', 'IMAGE', 'PDF', 'TEXT'):
        return JsonResponse({'error': f'Invalid output_type: {output_type}'}, status=400)
    if rating not in ('positive', 'negative'):
        return JsonResponse({'error': 'rating must be "positive" or "negative"'}, status=400)
    if not response_summary:
        return JsonResponse({'error': 'response_summary is required'}, status=400)

    # Truncate response_summary to keep DB lean
    feedback = Feedback.objects.create(
        query_text=query_text[:2000],
        output_type=output_type,
        response_summary=response_summary[:2000],
        rating=rating,
        comment=comment[:1000],
    )

    logger.info(f"Feedback #{feedback.pk} saved: [{rating}] {output_type} — {query_text[:80]}")
    return JsonResponse({
        'success': True,
        'feedback_id': feedback.pk,
        'message': 'Thank you for your feedback!'
    })


@csrf_exempt
@require_http_methods(["GET"])
def feedback_history(request):
    """
    GET /api/feedback/history/?limit=50&rating=positive&output_type=CSV
    Returns recent feedback entries (for admin/debugging).
    """
    limit = min(int(request.GET.get('limit', 50)), 200)
    qs = Feedback.objects.all()

    # Optional filters
    rating = request.GET.get('rating')
    if rating in ('positive', 'negative'):
        qs = qs.filter(rating=rating)

    output_type = (request.GET.get('output_type') or '').upper()
    if output_type in ('CSV', 'IMAGE', 'PDF', 'TEXT'):
        qs = qs.filter(output_type=output_type)

    entries = list(
        qs[:limit].values(
            'id', 'query_text', 'output_type', 'response_summary',
            'rating', 'comment', 'created_at'
        )
    )

    return JsonResponse({
        'count': len(entries),
        'results': entries
    })


@csrf_exempt
@require_http_methods(["GET"])
def feedback_stats(request):
    """
    GET /api/feedback/stats/
    Returns aggregate feedback statistics.
    """
    total = Feedback.objects.count()
    positive = Feedback.objects.filter(rating='positive').count()
    negative = Feedback.objects.filter(rating='negative').count()

    # Per output type
    by_type = {}
    for ot in ('CSV', 'IMAGE', 'PDF', 'TEXT'):
        qs = Feedback.objects.filter(output_type=ot)
        by_type[ot] = {
            'total': qs.count(),
            'positive': qs.filter(rating='positive').count(),
            'negative': qs.filter(rating='negative').count(),
        }

    return JsonResponse({
        'total': total,
        'positive': positive,
        'negative': negative,
        'by_output_type': by_type,
    })
