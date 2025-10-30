import pandas as pd
import numpy as np
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class AdvancedAIAnalyzer:
    def __init__(self):
        self.insight_templates = {
            'correlation': "Strong correlation ({correlation:.2f}) found between {col1} and {col2}",
            'trend': "Clear {trend} trend detected in {column} with strength {strength}",
            'anomaly': "Multiple anomalies ({count}) detected in {column} indicating potential data issues",
            'seasonality': "Seasonal pattern detected in {column} with period {period}",
            'cluster': "Data appears to form {clusters} distinct clusters in {columns}",
            'outlier': "Significant outliers detected in {column} affecting overall distribution"
        }
    
    def generate_ai_insights(self, df: pd.DataFrame, analysis_results: Dict) -> List[str]:
        """Generate intelligent insights from analysis results"""
        insights = []
        
        # Correlation insights
        if 'correlation_analysis' in analysis_results:
            corr_matrix = analysis_results['correlation_analysis']
            insights.extend(self._get_correlation_insights(corr_matrix))
        
        # Trend insights
        if 'trend_analysis' in analysis_results:
            trends = analysis_results['trend_analysis']
            insights.extend(self._get_trend_insights(trends))
        
        # Anomaly insights
        if 'anomaly_detection' in analysis_results:
            anomalies = analysis_results['anomaly_detection']
            insights.extend(self._get_anomaly_insights(anomalies))
        
        # Data quality insights
        insights.extend(self._get_data_quality_insights(df))
        
        return insights[:10]  # Return top 10 insights
    
    def predict_trends(self, df: pd.DataFrame, analysis_results: Dict) -> Dict[str, List[str]]:
        """Predict future trends based on current data"""
        predictions = {
            'short_term': [],
            'long_term': []
        }
        
        # Simple trend extrapolation for numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns
        
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            if len(df[col].dropna()) > 10:
                recent_trend = self._calculate_recent_trend(df[col])
                predictions['short_term'].append(
                    f"{col} expected to {recent_trend['direction']} by {recent_trend['magnitude']:.1f}% in short term"
                )
        
        return predictions
    
    def suggest_actions(self, df: pd.DataFrame, analysis_results: Dict, quality_issues: Dict) -> List[str]:
        """Suggest actionable recommendations"""
        actions = []
        
        # Data quality actions
        if quality_issues:
            actions.append("Address data quality issues before further analysis")
        
        # Correlation-based actions
        if 'correlation_analysis' in analysis_results:
            strong_corrs = self._find_strong_correlations(analysis_results['correlation_analysis'])
            for col1, col2, corr in strong_corrs:
                actions.append(f"Investigate relationship between {col1} and {col2} (correlation: {corr:.2f})")
        
        # Anomaly actions
        if 'anomaly_detection' in analysis_results:
            total_anomalies = sum(len(anom) for anom in analysis_results['anomaly_detection'].values())
            if total_anomalies > 0:
                actions.append(f"Review {total_anomalies} detected anomalies for data quality or business insights")
        
        return actions
    
    def data_storytelling(self, df: pd.DataFrame, analysis_results: Dict, ai_insights: List[str]) -> str:
        """Create a compelling data story"""
        story_parts = []
        
        # Introduction
        story_parts.append(f"This dataset contains {df.shape[0]:,} records across {df.shape[1]} features, revealing several interesting patterns.")
        
        # Key findings
        if ai_insights:
            story_parts.append("Key findings include:")
            for insight in ai_insights[:3]:
                story_parts.append(f"• {insight}")
        
        # Data quality note
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_pct > 5:
            story_parts.append(f"Note: The dataset has {missing_pct:.1f}% missing values that should be considered in analysis.")
        
        # Conclusion
        story_parts.append("These insights provide a foundation for data-driven decision making and further investigation.")
        
        return " ".join(story_parts)
    
    def _get_correlation_insights(self, corr_matrix: pd.DataFrame) -> List[str]:
        """Extract insights from correlation matrix"""
        insights = []
        n = len(corr_matrix.columns)
        
        for i in range(n):
            for j in range(i+1, n):
                corr = abs(corr_matrix.iloc[i, j])
                if corr > 0.7:  # Strong correlation
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    insights.append(
                        f"Strong correlation ({corr:.2f}) between {col1} and {col2}"
                    )
        
        return insights
    
    def _get_trend_insights(self, trends: Dict) -> List[str]:
        """Extract insights from trend analysis"""
        insights = []
        
        for col, trend_info in trends.items():
            if trend_info.get('strength', 0) > 0.5:  # Strong trend
                direction = trend_info.get('direction', 'unknown')
                insights.append(f"Strong {direction} trend in {col}")
        
        return insights
    
    def _get_anomaly_insights(self, anomalies: Dict) -> List[str]:
        """Extract insights from anomaly detection"""
        insights = []
        
        for col, anomaly_list in anomalies.items():
            if len(anomaly_list) > len(anomaly_list) * 0.05:  # More than 5% anomalies
                insights.append(f"High anomaly rate ({len(anomaly_list)}) in {col} requires investigation")
        
        return insights
    
    def _get_data_quality_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate data quality insights"""
        insights = []
        
        # Missing values insight
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_pct > 10:
            insights.append(f"High missing data rate ({missing_pct:.1f}%) may affect analysis reliability")
        
        # Constant columns insight
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            insights.append(f"Constant columns detected: {', '.join(constant_cols)}")
        
        return insights
    
    def _calculate_recent_trend(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate recent trend in a series"""
        clean_series = series.dropna()
        if len(clean_series) < 2:
            return {'direction': 'remain stable', 'magnitude': 0}
        
        # Simple linear trend calculation
        x = np.arange(len(clean_series))
        y = clean_series.values
        slope = np.polyfit(x, y, 1)[0]
        
        direction = "increase" if slope > 0 else "decrease"
        magnitude = abs(slope / np.mean(y)) * 100 if np.mean(y) != 0 else 0
        
        return {'direction': direction, 'magnitude': magnitude}
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[tuple]:
        """Find strong correlations above threshold"""
        strong_corrs = []
        n = len(corr_matrix.columns)
        
        for i in range(n):
            for j in range(i+1, n):
                corr = abs(corr_matrix.iloc[i, j])
                if corr > threshold:
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    strong_corrs.append((col1, col2, corr))
        
        return strong_corrs

# Singleton instance
ai_analyzer = AdvancedAIAnalyzer()

# Public functions
def generate_ai_insights(df, analysis_results):
    return ai_analyzer.generate_ai_insights(df, analysis_results)

def predict_trends(df, analysis_results):
    return ai_analyzer.predict_trends(df, analysis_results)

def suggest_actions(df, analysis_results, quality_issues):
    return ai_analyzer.suggest_actions(df, analysis_results, quality_issues)

def data_storytelling(df, analysis_results, ai_insights):
    return ai_analyzer.data_storytelling(df, analysis_results, ai_insights)
