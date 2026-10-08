"""Analysis orchestration.

Entry points live in :mod:`topspin_review.analysis.strategies` (``analyze`` and
``analyze_agentic``); the report-writing layer is :mod:`topspin_review.analysis.report`.
Shared run machinery is in :mod:`topspin_review.analysis.session`. Kept side-effect
free so importing the package stays cheap.
"""
