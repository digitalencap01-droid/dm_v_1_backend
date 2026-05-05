"""
PDF Report Generation Service for BizMentor AI

FIXES v2:
- All 17 sections from prompt structure now included (was 10 before)
- _parse_report_content() regex improved — handles ##, emojis, numbered headers
- Executive Dashboard now dynamically parses LLM content instead of hardcoded values
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import re
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT


class BizMentorPDFGenerator:
    """Generate professional PDF reports for BizMentor AI"""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report"""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#667eea')
        ))

        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=10,
            textColor=colors.HexColor('#667eea'),
            borderColor=colors.HexColor('#667eea'),
            borderWidth=1,
            borderPadding=5,
            borderRadius=3
        ))

        self.styles.add(ParagraphStyle(
            name='SubSectionHeader',
            parent=self.styles['Heading3'],
            fontSize=13,
            spaceAfter=8,
            spaceBefore=6,
            textColor=colors.HexColor('#444444'),
        ))

        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=13,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor('#4CAF50')
        ))

        self.styles.add(ParagraphStyle(
            name='WarningText',
            parent=self.styles['Normal'],
            textColor=colors.red,
            fontSize=11
        ))

        self.styles.add(ParagraphStyle(
            name='SuccessText',
            parent=self.styles['Normal'],
            textColor=colors.green,
            fontSize=11
        ))

        self.styles.add(ParagraphStyle(
            name='BottomLine',
            parent=self.styles['Normal'],
            fontSize=11,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor('#333333'),
            backColor=colors.HexColor('#f5f5f5'),
            borderPadding=6,
            spaceAfter=10,
        ))

        self.styles.add(ParagraphStyle(
            name='ConfidenceHigh',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#2e7d32'),
        ))

        self.styles.add(ParagraphStyle(
            name='ConfidenceMedium',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#e65100'),
        ))

        self.styles.add(ParagraphStyle(
            name='ConfidenceLow',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#c62828'),
        ))

    # -----------------------------------------------------------------------
    # FIX 1: Improved HTML cleaner — handles markdown symbols too
    # -----------------------------------------------------------------------

    def _clean_text(self, text: str) -> str:
        """Remove HTML tags, markdown symbols, and clean text for PDF"""
        if not text:
            return ""
        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', '', text)
        # Remove markdown bold/italic
        clean = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', clean)
        # Remove markdown headers (##, ###)
        clean = re.sub(r'^#{1,6}\s*', '', clean, flags=re.MULTILINE)
        # Remove emojis that reportlab cannot render
        clean = re.sub(
            r'[\U00010000-\U0010ffff'
            r'\U0001F300-\U0001F9FF'
            r'\u2600-\u27BF'
            r'\u2300-\u23FF'
            r'\u25A0-\u25FF'
            r'\u2700-\u27BF'
            r'\u2B00-\u2BFF'
            r'\uFE00-\uFE0F'
            r'\u200d'
            r'\u20e3'
            r']+',
            '', clean
        )
        # Clean markdown table separators
        clean = re.sub(r'\|[-: ]+\|[-| :]*', '', clean)
        # Clean extra whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean

    def _safe_paragraph(self, text: str, style) -> Optional[Paragraph]:
        """Safely create a paragraph, return None if text is empty"""
        cleaned = self._clean_text(text)
        if not cleaned:
            return None
        # Escape reportlab XML special chars
        cleaned = cleaned.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        try:
            return Paragraph(cleaned, style)
        except Exception:
            return None

    # -----------------------------------------------------------------------
    # Metadata section
    # -----------------------------------------------------------------------

    def _create_metadata_section(self, intake: Dict[str, Any]) -> List[Any]:
        """Create the report metadata section"""
        elements = []

        elements.append(Paragraph("BizMentor AI — Business Intelligence Report", self.styles['ReportTitle']))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#667eea')))
        elements.append(Spacer(1, 20))

        metadata_data = [
            ['Business:', intake.get('business_name', intake.get('business_idea', 'N/A'))],
            ['Report Mode:', intake.get('report_mode', 'Standard Plan')],
            ['Location:', intake.get('location', 'India')],
            ['Stage:', intake.get('stage', 'N/A')],
            ['Generated On:', datetime.now().strftime('%B %d, %Y — %H:%M')],
        ]

        metadata_table = Table(metadata_data, colWidths=[1.8 * inch, 4.5 * inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f4ff')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#667eea')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d0d0d0')),
        ]))

        elements.append(metadata_table)
        elements.append(Spacer(1, 25))
        return elements

    # -----------------------------------------------------------------------
    # FIX 2: Dynamic Executive Dashboard — reads LLM content
    # -----------------------------------------------------------------------

    def _extract_verdict_from_content(self, content: str) -> str:
        """Parse actual verdict from LLM report content"""
        patterns = [
            r'Decision:\s*(Go|Pivot|Wait|Avoid)',
            r'Final Verdict[:\s]+\*{0,2}(STRONG GO|GO|PROCEED WITH CAUTION|PIVOT NEEDED|WAIT|AVOID)\*{0,2}',
            r'Recommendation:\s*(STRONG GO|GO|PROCEED WITH CAUTION|PIVOT NEEDED|WAIT|AVOID)',
            r'\b(STRONG GO|PIVOT NEEDED)\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                verdict = match.group(1).upper().strip()
                mapping = {
                    'GO': 'STRONG GO',
                    'PIVOT': 'PIVOT NEEDED',
                }
                return mapping.get(verdict, verdict)
        return 'PROCEED WITH CAUTION'

    def _extract_viability_score(self, content: str) -> str:
        """Parse viability score from LLM report content"""
        patterns = [
            r'Viability Score[:\s]+(\d+(?:\.\d+)?)\s*/\s*10',
            r'Confidence Score[:\s]+(\d+(?:\.\d+)?)\s*/\s*10',
            r'Score[:\s]+(\d+(?:\.\d+)?)/10',
        ]
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return f"{match.group(1)}/10"
        return 'N/A'

    def _extract_critical_insights(self, content: str) -> List[str]:
        """Extract top strategic insights from LLM report content"""
        insights = []

        # Try to find Top 3 strategic moves section
        moves_match = re.search(
            r'Top 3 strategic moves?[:\s]+(.*?)(?=Top 3 risks?|##|\Z)',
            content, re.IGNORECASE | re.DOTALL
        )
        if moves_match:
            block = moves_match.group(1)
            bullets = re.findall(r'(?:^|\n)\s*[\-\*\d\.]+\s+(.+)', block)
            insights.extend([b.strip() for b in bullets[:3] if b.strip()])

        # Fallback: extract bullet points from executive summary
        if not insights:
            exec_match = re.search(
                r'Executive Summary(.*?)(?=##|Business Snapshot|\Z)',
                content, re.IGNORECASE | re.DOTALL
            )
            if exec_match:
                block = exec_match.group(1)
                bullets = re.findall(r'(?:^|\n)\s*[\-\*\d\.]+\s+(.+)', block)
                insights.extend([b.strip() for b in bullets[:5] if b.strip()])

        # Last fallback: generic
        if not insights:
            insights = [
                "See Executive Summary section for strategic overview.",
                "Review Competitive Landscape for market positioning.",
                "Refer to 90-Day Action Plan for immediate next steps.",
            ]

        return insights[:5]

    def _create_executive_dashboard(self, report_data: Dict[str, Any]) -> List[Any]:
        """Create executive intelligence dashboard — dynamically from LLM content"""
        elements = []

        elements.append(Paragraph("Executive Intelligence Dashboard", self.styles['SectionHeader']))
        elements.append(Spacer(1, 10))

        # Dynamic values from parsed content
        viability = report_data.get('viability_score', 'N/A')
        verdict = report_data.get('verdict', 'PROCEED WITH CAUTION')
        competition = report_data.get('competition_level', 'See Competitive Landscape section')
        risk = report_data.get('risk_level', 'See Risk Flags section')

        metrics_data = [
            ['Metric', 'Value'],
            ['Viability Score', viability],
            ['Final Verdict', verdict],
            ['Competition Level', competition],
            ['Risk Level', risk],
            ['Report Generated', datetime.now().strftime('%d %b %Y')],
        ]

        metrics_table = Table(metrics_data, colWidths=[2.8 * inch, 3.5 * inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#f0f4ff')),
            ('TEXTCOLOR', (0, 1), (0, -1), colors.HexColor('#667eea')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
            ('TOPPADDING', (0, 0), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(metrics_table)
        elements.append(Spacer(1, 18))

        # Top insights
        elements.append(Paragraph("Top Strategic Insights", self.styles['SubSectionHeader']))
        insights = report_data.get('critical_insights', [])
        for i, insight in enumerate(insights[:5], 1):
            p = self._safe_paragraph(f"{i}. {insight}", self.styles['Normal'])
            if p:
                elements.append(p)
                elements.append(Spacer(1, 5))

        elements.append(Spacer(1, 15))

        # Verdict badge
        verdict_color_map = {
            'STRONG GO': colors.HexColor('#2e7d32'),
            'GO': colors.HexColor('#2e7d32'),
            'PROCEED WITH CAUTION': colors.HexColor('#e65100'),
            'WAIT': colors.HexColor('#1565c0'),
            'PIVOT NEEDED': colors.HexColor('#c62828'),
            'AVOID': colors.HexColor('#b71c1c'),
        }
        verdict_color = verdict_color_map.get(verdict.upper(), colors.HexColor('#e65100'))

        verdict_style = ParagraphStyle(
            'VerdictStyle',
            parent=self.styles['Normal'],
            fontSize=15,
            textColor=verdict_color,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            spaceAfter=20,
        )
        elements.append(Paragraph(f"Decision: {verdict}", verdict_style))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e0e0e0')))
        elements.append(Spacer(1, 20))

        return elements

    # -----------------------------------------------------------------------
    # Generic section content renderer
    # -----------------------------------------------------------------------

    def _create_section_content(self, section_title: str, content: str) -> List[Any]:
        """Create a standard section with title and content"""
        elements = []

        elements.append(Paragraph(section_title, self.styles['SectionHeader']))
        elements.append(Spacer(1, 8))

        if not content or not content.strip():
            elements.append(Paragraph(
                "Evidence is limited here — refer to the research dossier for more detail.",
                self.styles['WarningText']
            ))
            elements.append(Spacer(1, 15))
            return elements

        # Split on double newlines for paragraphs
        blocks = re.split(r'\n{2,}', content.strip())
        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Detect sub-headers (lines starting with ### or **)
            if re.match(r'^(#{2,4}|\*\*).{3,}', block):
                clean_header = self._clean_text(block)
                if clean_header:
                    p = self._safe_paragraph(clean_header, self.styles['SubSectionHeader'])
                    if p:
                        elements.append(p)
                        elements.append(Spacer(1, 4))
                continue

            # Detect markdown table rows
            if '|' in block and re.search(r'\|.+\|', block):
                table_elements = self._render_markdown_table(block)
                if table_elements:
                    elements.extend(table_elements)
                    elements.append(Spacer(1, 8))
                continue

            # Detect bullet list block
            bullet_lines = re.findall(r'(?:^|\n)\s*[-\*]\s+(.+)', block)
            numbered_lines = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s+(.+)', block)

            if bullet_lines:
                for item in bullet_lines:
                    p = self._safe_paragraph(f"• {item.strip()}", self.styles['Normal'])
                    if p:
                        elements.append(p)
                        elements.append(Spacer(1, 3))
                elements.append(Spacer(1, 6))
                continue

            if numbered_lines:
                for i, item in enumerate(numbered_lines, 1):
                    p = self._safe_paragraph(f"{i}. {item.strip()}", self.styles['Normal'])
                    if p:
                        elements.append(p)
                        elements.append(Spacer(1, 3))
                elements.append(Spacer(1, 6))
                continue

            # Bottom line highlight
            if re.match(r'^Bottom line[:\s]', block, re.IGNORECASE):
                p = self._safe_paragraph(block, self.styles['BottomLine'])
                if p:
                    elements.append(p)
                    elements.append(Spacer(1, 8))
                continue

            # Regular paragraph
            p = self._safe_paragraph(block, self.styles['Normal'])
            if p:
                elements.append(p)
                elements.append(Spacer(1, 6))

        elements.append(Spacer(1, 12))
        return elements

    def _render_markdown_table(self, block: str) -> List[Any]:
        """Convert markdown table text to reportlab Table"""
        try:
            lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
            # Filter out separator lines like |---|---|
            rows = []
            for line in lines:
                if re.match(r'^\|[-: |]+\|$', line):
                    continue
                cells = [self._clean_text(c.strip()) for c in line.strip('|').split('|')]
                if cells:
                    rows.append(cells)

            if not rows:
                return []

            max_cols = max(len(r) for r in rows)
            # Pad rows to equal length
            rows = [r + [''] * (max_cols - len(r)) for r in rows]

            col_width = 6.3 * inch / max_cols
            table = Table(rows, colWidths=[col_width] * max_cols, repeatRows=1)

            style = [
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [colors.white, colors.HexColor('#f9f9ff')]),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('WORDWRAP', (0, 0), (-1, -1), True),
            ]
            table.setStyle(TableStyle(style))
            return [table]
        except Exception:
            return []

    # -----------------------------------------------------------------------
    # FIX 3: All 17 sections now in generate_pdf_report
    # -----------------------------------------------------------------------

    def generate_pdf_report(
        self,
        intake: Dict[str, Any],
        report_content: str,
        sources: List[Any],
        output_path: str
    ) -> str:
        """Generate the complete PDF report — all 17 sections"""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=65,
            leftMargin=65,
            topMargin=65,
            bottomMargin=65
        )

        elements = []

        # Parse LLM report content
        report_data = self._parse_report_content(report_content)
        report_data['sources'] = sources

        # Cover + metadata
        elements.extend(self._create_metadata_section(intake))

        # Executive dashboard (dynamic)
        elements.extend(self._create_executive_dashboard(report_data))
        elements.append(PageBreak())

        # ---------------------------------------------------------------
        # ALL 17 SECTIONS — mapped to parsed keys
        # ---------------------------------------------------------------
        sections = [
            ("1. Business Snapshot",
             report_data.get('business_snapshot', '')),

            ("2. Demand & Trend Research",
             report_data.get('demand_trend', '')),

            ("3. Competitive Landscape",
             report_data.get('competitive_analysis', '')),

            ("4. Audience & Pain Point Research (ICP)",
             report_data.get('customer_insights', '')),

            ("5. People Reviews & Sentiment Analysis",
             report_data.get('people_reviews', '')),

            ("6. Marketplace & Pricing",
             report_data.get('pricing_revenue', '')),

            ("7. Ads Intensity & Marketing Spend",
             report_data.get('ads_intensity', '')),

            ("8. SEO & Search Opportunity",
             report_data.get('seo_search', '')),

            ("9. Local Feasibility Research",
             report_data.get('local_feasibility', '')),

            ("10. Founder Fit & Execution Feasibility",
             report_data.get('founder_fit', '')),

            ("11. Strategic Recommendation Synthesis",
             report_data.get('strategic_synthesis', '')),

            ("12. 90-Day Action Plan",
             report_data.get('action_plan', '')),

            ("13. Financial Projections",
             report_data.get('financial_projections', '')),

            ("14. Operations / Execution Considerations",
             report_data.get('operations_execution', '')),

            ("15. Marketing & Growth Recommendations",
             report_data.get('marketing_channels', '')),

            ("16. Legal / Compliance / Risk Flags",
             report_data.get('legal_compliance', '')),

            ("17. Final Verdict & Next Steps",
             report_data.get('final_verdict', '')),
        ]

        # If parser could not map most sections, do not lose report data.
        # Render the full raw AI output in dedicated sections so PDF always
        # carries the complete narrative.
        filled_sections = sum(1 for _, c in sections if c and c.strip())
        if filled_sections <= 2 and report_content and report_content.strip():
            raw_text = report_content.strip()
            sections = [
                ("1. Full Research Narrative", raw_text),
                ("2. Extracted Strategic Highlights", report_data.get('executive_summary', '')),
                ("3. Final Verdict & Next Steps", report_data.get('final_verdict', raw_text)),
            ]

        for section_title, section_content in sections:
            elements.extend(self._create_section_content(section_title, section_content))
            elements.append(Spacer(1, 10))
            elements.append(HRFlowable(width="100%", thickness=0.7, color=colors.HexColor('#e6e6e6')))
            elements.append(Spacer(1, 10))

        # Sources section
        if report_data.get('sources'):
            elements.append(Paragraph("Sources Consulted", self.styles['SectionHeader']))
            elements.append(Spacer(1, 10))
            for source in report_data['sources'][:20]:
                if isinstance(source, dict):
                    title = source.get('title', 'Unknown')
                    url = source.get('url', '')
                    source_text = f"• {title} — {url}" if url else f"• {title}"
                else:
                    source_text = f"• {str(source)}"
                p = self._safe_paragraph(source_text, self.styles['Normal'])
                if p:
                    elements.append(p)
                    elements.append(Spacer(1, 3))

        doc.build(elements)
        return output_path

    # -----------------------------------------------------------------------
    # FIX 4: Improved parser — handles ##, emojis, numbered headers, variants
    # -----------------------------------------------------------------------

    def _parse_report_content(self, content: str) -> Dict[str, Any]:
        """
        Parse LLM-generated report content into structured sections.
        Handles: ## headers, emoji prefixes, numbered sections, case variants.
        """
        if not content:
            return self._default_metadata()

        # Section header patterns → dict key mapping
        # Each entry: (regex_pattern, dict_key)
        section_patterns = [
            (r'business snapshot', 'business_snapshot'),
            (r'demand\s*[&and]*\s*trend', 'demand_trend'),
            (r'market demand', 'demand_trend'),
            (r'competitive\s*landscape', 'competitive_analysis'),
            (r'competitor', 'competitive_analysis'),
            (r'audience\s*[&and]*\s*pain', 'customer_insights'),
            (r'customer\s*[&and/or]*\s*icp', 'customer_insights'),
            (r'icp\s*insights', 'customer_insights'),
            (r'people\s*reviews', 'people_reviews'),
            (r'sentiment\s*analysis', 'people_reviews'),
            (r'review\s*[&and]*\s*sentiment', 'people_reviews'),
            (r'marketplace\s*[&and]*\s*pricing', 'pricing_revenue'),
            (r'pricing\s*/\s*revenue', 'pricing_revenue'),
            (r'pricing.*unit.?economics', 'pricing_revenue'),
            (r'ads\s*intensity', 'ads_intensity'),
            (r'marketing\s*spend', 'ads_intensity'),
            (r'seo\s*[&and]*\s*search', 'seo_search'),
            (r'search\s*opportunity', 'seo_search'),
            (r'local\s*feasibility', 'local_feasibility'),
            (r'founder\s*fit', 'founder_fit'),
            (r'execution\s*feasibility', 'founder_fit'),
            (r'strategic\s*recommendation', 'strategic_synthesis'),
            (r'recommendation\s*synthesis', 'strategic_synthesis'),
            (r'go.to.market', 'strategic_synthesis'),
            (r'90.day\s*action', 'action_plan'),
            (r'action\s*plan', 'action_plan'),
            (r'financial\s*projection', 'financial_projections'),
            (r'p&l|profit\s*[&and]*\s*loss', 'financial_projections'),
            (r'operations\s*/\s*execution', 'operations_execution'),
            (r'operations.*consideration', 'operations_execution'),
            (r'marketing.*growth.*recommendation', 'marketing_channels'),
            (r'growth\s*recommendation', 'marketing_channels'),
            (r'legal\s*/\s*compliance', 'legal_compliance'),
            (r'risk\s*flags', 'legal_compliance'),
            (r'compliance.*risk', 'legal_compliance'),
            (r'final\s*verdict', 'final_verdict'),
            (r'next\s*steps', 'final_verdict'),
            (r'sources\s*consulted', 'sources_text'),
            (r'executive\s*(summary|intelligence|dashboard)', 'executive_summary'),
        ]

        sections: Dict[str, str] = {}
        current_key: Optional[str] = None
        current_lines: List[str] = []

        def _flush():
            if current_key and current_lines:
                block = '\n'.join(current_lines).strip()
                if block:
                    # Append if key already exists (section appeared twice)
                    if current_key in sections:
                        sections[current_key] += '\n\n' + block
                    else:
                        sections[current_key] = block

        for raw_line in content.split('\n'):
            line = raw_line.strip()

            # Detect section header: starts with #, number+dot, or ALL CAPS line
            is_header = bool(
                re.match(r'^#{1,4}\s+', line) or
                re.match(r'^\d{1,2}[\.\)]\s+[A-Z]', line) or
                (len(line) > 4 and line.isupper() and len(line) < 80)
            )

            if is_header:
                # Clean the line to get plain text for matching
                plain = re.sub(r'^#{1,4}\s*', '', line)
                plain = re.sub(r'^\d{1,2}[\.\)]\s*', '', plain)
                plain = re.sub(
                    r'[\U00010000-\U0010ffff\U0001F300-\U0001F9FF\u2600-\u27BF]+',
                    '', plain
                ).strip()

                matched_key = None
                for pattern, key in section_patterns:
                    if re.search(pattern, plain, re.IGNORECASE):
                        matched_key = key
                        break

                if matched_key:
                    _flush()
                    current_key = matched_key
                    current_lines = []
                    continue
                # else: unrecognized header, treat as content

            if current_key is not None:
                current_lines.append(raw_line)

        _flush()

        # Add dynamic metadata from content
        meta = self._default_metadata()
        meta['viability_score'] = self._extract_viability_score(content)
        meta['verdict'] = self._extract_verdict_from_content(content)
        meta['critical_insights'] = self._extract_critical_insights(content)

        # Merge parsed sections into meta
        meta.update(sections)

        return meta

    def _default_metadata(self) -> Dict[str, Any]:
        """Return default metadata values"""
        return {
            'viability_score': 'N/A',
            'verdict': 'PROCEED WITH CAUTION',
            'competition_level': 'See Competitive Landscape section',
            'risk_level': 'See Legal / Risk Flags section',
            'critical_insights': [
                "Refer to Executive Summary for strategic overview.",
                "Review Competitive Landscape for market positioning.",
                "Refer to 90-Day Action Plan for immediate next steps.",
            ],
            # Section placeholders
            'business_snapshot': '',
            'demand_trend': '',
            'competitive_analysis': '',
            'customer_insights': '',
            'people_reviews': '',
            'pricing_revenue': '',
            'ads_intensity': '',
            'seo_search': '',
            'local_feasibility': '',
            'founder_fit': '',
            'strategic_synthesis': '',
            'action_plan': '',
            'financial_projections': '',
            'operations_execution': '',
            'marketing_channels': '',
            'legal_compliance': '',
            'final_verdict': '',
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def generate_bizmentor_pdf(
    intake: Dict[str, Any],
    report_content: str,
    sources: List[Any] = None,
    output_filename: str = None
) -> str:
    """Convenience function to generate BizMentor PDF report"""
    if output_filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        raw_name = intake.get('business_name', intake.get('business_idea', 'business'))
        safe_name = re.sub(r'[^\w\-]', '_', str(raw_name))[:40]
        output_filename = f"bizmentor_report_{safe_name}_{timestamp}.pdf"

    output_path = Path(__file__).resolve().parents[3] / "reports" / output_filename
    output_path.parent.mkdir(exist_ok=True)

    generator = BizMentorPDFGenerator()
    return generator.generate_pdf_report(
        intake,
        report_content,
        sources or [],
        str(output_path)
    )
