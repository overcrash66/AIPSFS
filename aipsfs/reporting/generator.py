# reporter.py
import os
import logging
from datetime import datetime
from typing import List, Dict, Tuple
import pandas as pd
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

class ReportGenerator:
    """Generates PDF reports for stock analysis results."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_report(self, top_stocks: List[Dict], 
                      analysis_period: Tuple[datetime, datetime]) -> str:
        """Generate a comprehensive PDF report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stock_report_{timestamp}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # Cover page
            self._add_cover_page(pdf, analysis_period)
            
            # Summary table
            self._add_summary_table(pdf, top_stocks)
            
            # Detailed analysis for each stock
            for stock in top_stocks:
                self._add_stock_analysis(pdf, stock)
            
            # Save the PDF
            pdf.output(filepath)
            logging.info(f"Report generated successfully: {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"Report generation failed: {str(e)}")
            raise
    
    def _add_cover_page(self, pdf: FPDF, analysis_period: Tuple[datetime, datetime]):
        """Add a cover page to the report."""
        pdf.add_page()
        pdf.set_font('Arial', 'B', 28)
        pdf.cell(0, 40, "AI-Powered Stock Forecast Report", 0, 1, 'C')
        
        pdf.set_font('Arial', '', 18)
        pdf.cell(0, 20, "Predicting Future Returns with Machine Learning", 0, 1, 'C')
        
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 100, f"Analysis Period: {analysis_period[0].strftime('%Y-%m-%d')} to {analysis_period[1].strftime('%Y-%m-%d')}", 0, 1, 'C')
        pdf.cell(0, 20, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
        
        pdf.set_font('Arial', 'I', 10)
        disclaimer = (
            "Disclaimer: This report is for informational purposes only and does not constitute financial advice. "
            "Stock prices are volatile, and past performance is not indicative of future results."
        )
        pdf.multi_cell(0, 5, disclaimer, 0, 'C')
    
    def _add_summary_table(self, pdf: FPDF, top_stocks: List[Dict]):
        """Add a summary table of top stocks."""
        pdf.add_page()
        pdf.set_font('Arial', 'B', 18)
        pdf.cell(0, 10, "Top Performing Stocks Summary", 0, 1)
        pdf.ln(5)
        
        # Table headers
        pdf.set_font('Arial', 'B', 10)
        col_widths = [25, 60, 28, 28, 28]
        headers = ['Symbol', 'Company Name', 'Current Price', 'Target Price', 'Expected Return (%)']
        
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 10, header, 1, 0, 'C')
        pdf.ln()
        
        # Table data
        pdf.set_font('Arial', '', 10)
        for stock in top_stocks:
            pdf.cell(col_widths[0], 10, stock['symbol'], 1, 0, 'C')
            pdf.cell(col_widths[1], 10, stock['name'][:30] + '...' if len(stock['name']) > 30 else stock['name'], 1, 0, 'L')
            pdf.cell(col_widths[2], 10, f"${stock['current_price']:.2f}", 1, 0, 'C')
            pdf.cell(col_widths[3], 10, f"${stock['predicted_price']:.2f}", 1, 0, 'C')
            pdf.cell(col_widths[4], 10, f"{stock['return_pct']:.2f}%", 1, 0, 'C')
            pdf.ln()
    
    def _add_stock_analysis(self, pdf: FPDF, stock: Dict):
        """Add detailed analysis for a single stock."""
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, f"{stock['symbol']} - {stock['name']}", 0, 1)
        pdf.ln(5)
        
        # Key metrics
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Key Metrics:", 0, 1)
        
        pdf.set_font('Arial', '', 11)
        metrics_text = (
            f"Current Price: ${stock['current_price']:.2f}\n"
            f"Predicted Price: ${stock['predicted_price']:.2f}\n"
            f"Expected Return: {stock['return_pct']:.2f}%\n"
            f"Model R²: {stock['model_metrics']['r2']:.4f}\n"
            f"Directional Accuracy: {stock['model_metrics']['directional_accuracy']:.2f}%"
        )
        pdf.multi_cell(0, 8, metrics_text)
        pdf.ln(5)
        
        # Add chart
        chart_path = self._create_price_chart(stock)
        if chart_path:
            pdf.image(chart_path, w=180, h=100)
            os.remove(chart_path)  # Clean up temporary file
    
    def _create_price_chart(self, stock: Dict) -> str:
        """Create a price chart for the stock."""
        try:
            # Combine historical and forecast data
            historical_dates = pd.to_datetime(stock['historical_dates'])
            forecast_dates = pd.to_datetime(stock['forecast_dates'])
            
            plt.figure(figsize=(12, 6))
            
            # Plot historical prices
            plt.plot(historical_dates, stock['historical_prices'], 
                    label='Historical Prices', color='blue', linewidth=2)
            
            # Plot forecast prices
            plt.plot(forecast_dates, stock['forecast_prices'], 
                    label='Forecast Prices', color='red', linestyle='--', linewidth=2)
            
            # Add vertical line at forecast start
            plt.axvline(x=historical_dates[-1], color='gray', linestyle=':', alpha=0.7)
            
            # Formatting
            plt.title(f"{stock['symbol']} Price Forecast", fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price ($)', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save to temporary file
            temp_path = f"temp_{stock['symbol']}_chart.png"
            plt.savefig(temp_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return temp_path
            
        except Exception as e:
            logging.error(f"Failed to create chart for {stock['symbol']}: {str(e)}")
            return None