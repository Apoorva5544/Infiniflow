"""
Analytics and Monitoring System
- Usage tracking
- Performance metrics
- Cost analysis
- User behavior analytics
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from . import models
import json


class AnalyticsEngine:
    """Comprehensive analytics for the AI platform"""
    
    @staticmethod
    def get_workspace_analytics(
        db: Session, 
        workspace_id: int,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive workspace analytics"""
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Query statistics
        total_queries = db.query(models.QueryLog).filter(
            and_(
                models.QueryLog.workspace_id == workspace_id,
                models.QueryLog.created_at >= start_date
            )
        ).count()
        
        # Average latency
        avg_latency = db.query(
            func.avg(models.QueryLog.latency_ms)
        ).filter(
            and_(
                models.QueryLog.workspace_id == workspace_id,
                models.QueryLog.created_at >= start_date
            )
        ).scalar() or 0
        
        # Total tokens used
        total_tokens = db.query(
            func.sum(models.QueryLog.tokens_used)
        ).filter(
            and_(
                models.QueryLog.workspace_id == workspace_id,
                models.QueryLog.created_at >= start_date
            )
        ).scalar() or 0
        
        # Success rate
        successful_queries = db.query(models.QueryLog).filter(
            and_(
                models.QueryLog.workspace_id == workspace_id,
                models.QueryLog.status == "completed",
                models.QueryLog.created_at >= start_date
            )
        ).count()
        
        success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
        
        # Document count
        doc_count = db.query(models.Document).filter(
            models.Document.workspace_id == workspace_id
        ).count()
        
        # Query trends (daily)
        daily_queries = db.query(
            func.date(models.QueryLog.created_at).label('date'),
            func.count(models.QueryLog.id).label('count')
        ).filter(
            and_(
                models.QueryLog.workspace_id == workspace_id,
                models.QueryLog.created_at >= start_date
            )
        ).group_by(func.date(models.QueryLog.created_at)).all()
        
        # Most used models
        model_usage = db.query(
            models.QueryLog.model_used,
            func.count(models.QueryLog.id).label('count')
        ).filter(
            and_(
                models.QueryLog.workspace_id == workspace_id,
                models.QueryLog.created_at >= start_date
            )
        ).group_by(models.QueryLog.model_used).all()
        
        return {
            "period_days": days,
            "total_queries": total_queries,
            "avg_latency_ms": round(avg_latency, 2),
            "total_tokens": int(total_tokens),
            "success_rate": round(success_rate, 2),
            "document_count": doc_count,
            "daily_trends": [
                {"date": str(date), "queries": count}
                for date, count in daily_queries
            ],
            "model_usage": [
                {"model": model, "count": count}
                for model, count in model_usage
            ],
            "estimated_cost_usd": AnalyticsEngine.estimate_cost(total_tokens)
        }
    
    @staticmethod
    def get_user_analytics(
        db: Session,
        user_id: int,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get user-level analytics"""
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Total queries
        total_queries = db.query(models.QueryLog).filter(
            and_(
                models.QueryLog.user_id == user_id,
                models.QueryLog.created_at >= start_date
            )
        ).count()
        
        # Workspaces accessed
        workspaces = db.query(models.Workspace).filter(
            models.Workspace.owner_id == user_id
        ).all()
        
        # API usage
        api_calls = db.query(models.APIUsage).filter(
            and_(
                models.APIUsage.user_id == user_id,
                models.APIUsage.created_at >= start_date
            )
        ).count()
        
        # Average response time
        avg_response_time = db.query(
            func.avg(models.APIUsage.response_time_ms)
        ).filter(
            and_(
                models.APIUsage.user_id == user_id,
                models.APIUsage.created_at >= start_date
            )
        ).scalar() or 0
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_queries": total_queries,
            "workspace_count": len(workspaces),
            "api_calls": api_calls,
            "avg_response_time_ms": round(avg_response_time, 2),
            "workspaces": [
                {
                    "id": ws.id,
                    "name": ws.name,
                    "documents": ws.total_documents,
                    "queries": ws.total_queries
                }
                for ws in workspaces
            ]
        }
    
    @staticmethod
    def get_platform_analytics(
        db: Session,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get platform-wide analytics"""
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Total users
        total_users = db.query(models.User).count()
        active_users = db.query(models.User).filter(
            models.User.last_login >= start_date
        ).count()
        
        # Total workspaces
        total_workspaces = db.query(models.Workspace).count()
        
        # Total documents
        total_documents = db.query(models.Document).count()
        
        # Total queries
        total_queries = db.query(models.QueryLog).filter(
            models.QueryLog.created_at >= start_date
        ).count()
        
        # Total tokens
        total_tokens = db.query(
            func.sum(models.QueryLog.tokens_used)
        ).filter(
            models.QueryLog.created_at >= start_date
        ).scalar() or 0
        
        # Average latency
        avg_latency = db.query(
            func.avg(models.QueryLog.latency_ms)
        ).filter(
            models.QueryLog.created_at >= start_date
        ).scalar() or 0
        
        return {
            "period_days": days,
            "total_users": total_users,
            "active_users": active_users,
            "total_workspaces": total_workspaces,
            "total_documents": total_documents,
            "total_queries": total_queries,
            "total_tokens": int(total_tokens),
            "avg_latency_ms": round(avg_latency, 2),
            "estimated_monthly_cost_usd": AnalyticsEngine.estimate_cost(total_tokens * 30 / days)
        }
    
    @staticmethod
    def estimate_cost(tokens: int) -> float:
        """
        Estimate API cost based on token usage
        Using approximate pricing for Groq/OpenAI
        """
        # Groq pricing (approximate): $0.10 per 1M tokens
        cost_per_million = 0.10
        return round((tokens / 1_000_000) * cost_per_million, 4)
    
    @staticmethod
    def get_query_insights(
        db: Session,
        workspace_id: int,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Get insights about query patterns"""
        
        # Slowest queries
        slowest = db.query(models.QueryLog).filter(
            models.QueryLog.workspace_id == workspace_id
        ).order_by(models.QueryLog.latency_ms.desc()).limit(limit).all()
        
        # Most token-intensive queries
        token_intensive = db.query(models.QueryLog).filter(
            models.QueryLog.workspace_id == workspace_id
        ).order_by(models.QueryLog.tokens_used.desc()).limit(limit).all()
        
        # Recent failed queries
        failed = db.query(models.QueryLog).filter(
            and_(
                models.QueryLog.workspace_id == workspace_id,
                models.QueryLog.status == "failed"
            )
        ).order_by(models.QueryLog.created_at.desc()).limit(limit).all()
        
        return {
            "slowest_queries": [
                {
                    "query": q.query_text[:100],
                    "latency_ms": q.latency_ms,
                    "timestamp": q.created_at.isoformat()
                }
                for q in slowest
            ],
            "token_intensive_queries": [
                {
                    "query": q.query_text[:100],
                    "tokens": q.tokens_used,
                    "timestamp": q.created_at.isoformat()
                }
                for q in token_intensive
            ],
            "failed_queries": [
                {
                    "query": q.query_text[:100],
                    "error": q.response_text[:200] if q.response_text else "Unknown error",
                    "timestamp": q.created_at.isoformat()
                }
                for q in failed
            ]
        }
    
    @staticmethod
    def export_analytics(
        db: Session,
        workspace_id: int,
        format: str = "json"
    ) -> str:
        """Export analytics data"""
        
        analytics = AnalyticsEngine.get_workspace_analytics(db, workspace_id, days=90)
        insights = AnalyticsEngine.get_query_insights(db, workspace_id)
        
        export_data = {
            "workspace_id": workspace_id,
            "exported_at": datetime.utcnow().isoformat(),
            "analytics": analytics,
            "insights": insights
        }
        
        if format == "json":
            return json.dumps(export_data, indent=2)
        else:
            # Could add CSV, PDF export here
            return json.dumps(export_data, indent=2)
