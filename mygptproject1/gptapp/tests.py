import json
from django.test import TestCase, Client
from .models import Feedback


class FeedbackAPITest(TestCase):
    """Test the feedback collection and retrieval system."""

    def setUp(self):
        self.client = Client()

    def test_submit_positive_feedback(self):
        """Submit positive feedback and verify it's stored."""
        resp = self.client.post(
            '/api/feedback/',
            data=json.dumps({
                'query_text': 'Analyze survival rates by age group',
                'output_type': 'CSV',
                'response_summary': 'Generated Python code with Kaplan-Meier analysis...',
                'rating': 'positive',
                'comment': 'Great analysis, very detailed',
            }),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data['success'])
        self.assertIn('feedback_id', data)

        # Verify in DB
        fb = Feedback.objects.get(pk=data['feedback_id'])
        self.assertEqual(fb.rating, 'positive')
        self.assertEqual(fb.output_type, 'CSV')
        self.assertIn('survival rates', fb.query_text)

    def test_submit_negative_feedback_with_comment(self):
        """Submit negative feedback with improvement suggestions."""
        resp = self.client.post(
            '/api/feedback/',
            data=json.dumps({
                'query_text': 'Generate PDF report',
                'output_type': 'PDF',
                'response_summary': 'Agent A code generated...',
                'rating': 'negative',
                'comment': 'Code had syntax errors in matplotlib section',
            }),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 200)
        fb = Feedback.objects.last()
        self.assertEqual(fb.rating, 'negative')
        self.assertIn('syntax errors', fb.comment)

    def test_submit_feedback_validation(self):
        """Reject feedback with missing required fields."""
        # Missing query_text
        resp = self.client.post(
            '/api/feedback/',
            data=json.dumps({
                'output_type': 'CSV',
                'response_summary': 'test',
                'rating': 'positive',
            }),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)

        # Invalid rating
        resp = self.client.post(
            '/api/feedback/',
            data=json.dumps({
                'query_text': 'test',
                'output_type': 'CSV',
                'response_summary': 'test',
                'rating': 'neutral',
            }),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)

        # Invalid output_type
        resp = self.client.post(
            '/api/feedback/',
            data=json.dumps({
                'query_text': 'test',
                'output_type': 'EXCEL',
                'response_summary': 'test',
                'rating': 'positive',
            }),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)

    def test_feedback_history(self):
        """Retrieve feedback history with filters."""
        Feedback.objects.create(
            query_text='Q1', output_type='CSV',
            response_summary='R1', rating='positive',
        )
        Feedback.objects.create(
            query_text='Q2', output_type='PDF',
            response_summary='R2', rating='negative', comment='Bad output',
        )
        Feedback.objects.create(
            query_text='Q3', output_type='CSV',
            response_summary='R3', rating='positive',
        )

        # All feedback
        resp = self.client.get('/api/feedback/history/')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data['count'], 3)

        # Filter by rating
        resp = self.client.get('/api/feedback/history/?rating=positive')
        data = resp.json()
        self.assertEqual(data['count'], 2)

        # Filter by output_type
        resp = self.client.get('/api/feedback/history/?output_type=PDF')
        data = resp.json()
        self.assertEqual(data['count'], 1)

    def test_feedback_stats(self):
        """Verify aggregate statistics endpoint."""
        Feedback.objects.create(
            query_text='Q1', output_type='CSV',
            response_summary='R1', rating='positive',
        )
        Feedback.objects.create(
            query_text='Q2', output_type='CSV',
            response_summary='R2', rating='negative',
        )

        resp = self.client.get('/api/feedback/stats/')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data['total'], 2)
        self.assertEqual(data['positive'], 1)
        self.assertEqual(data['negative'], 1)
        self.assertEqual(data['by_output_type']['CSV']['total'], 2)


class FeedbackPromptEnrichmentTest(TestCase):
    """Test feedback-to-prompt pipeline."""

    def test_empty_feedback_returns_empty_string(self):
        """No feedback produces no context."""
        from .gpt_backend_utils import build_feedback_context
        result = build_feedback_context('CSV')
        self.assertEqual(result, '')

    def test_positive_feedback_generates_context(self):
        """Positive feedback produces guidance section."""
        from .gpt_backend_utils import build_feedback_context

        Feedback.objects.create(
            query_text='Analyze patient demographics',
            output_type='CSV',
            response_summary='Used pandas groupby for age distribution, seaborn for visualization',
            rating='positive',
            comment='Clear and accurate',
        )

        result = build_feedback_context('CSV')
        self.assertIn('Guidance from Past Successful Analyses', result)
        self.assertIn('patient demographics', result)
        self.assertIn('pandas groupby', result)

    def test_negative_feedback_generates_warnings(self):
        """Negative feedback with comments produces warnings."""
        from .gpt_backend_utils import build_feedback_context

        Feedback.objects.create(
            query_text='Generate chart',
            output_type='IMAGE',
            response_summary='matplotlib bar chart',
            rating='negative',
            comment='Chart labels were overlapping and unreadable',
        )

        result = build_feedback_context('IMAGE')
        self.assertIn('Things to Avoid', result)
        self.assertIn('overlapping', result)

    def test_feedback_scoped_by_output_type(self):
        """Feedback from CSV doesn't leak into PDF context."""
        from .gpt_backend_utils import build_feedback_context

        Feedback.objects.create(
            query_text='CSV analysis', output_type='CSV',
            response_summary='CSV code', rating='positive',
        )

        result = build_feedback_context('PDF')
        self.assertEqual(result, '')

    def test_mixed_feedback_context(self):
        """Both positive and negative feedback combine properly."""
        from .gpt_backend_utils import build_feedback_context

        Feedback.objects.create(
            query_text='Good query', output_type='PDF',
            response_summary='Good response', rating='positive',
        )
        Feedback.objects.create(
            query_text='Bad query', output_type='PDF',
            response_summary='Bad response', rating='negative',
            comment='Too slow and inaccurate',
        )

        result = build_feedback_context('PDF')
        self.assertIn('Guidance from Past Successful Analyses', result)
        self.assertIn('Things to Avoid', result)
        self.assertIn('Too slow', result)
