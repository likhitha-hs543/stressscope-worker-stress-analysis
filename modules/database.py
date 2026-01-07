"""
Database Models
Defines data structure for session tracking, stress records, and analytics
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from config import DATABASE_URL

Base = declarative_base()


class Employee(Base):
    """
    Employee/User table
    Stores minimal identifying information
    """
    __tablename__ = 'employees'
    
    id = Column(Integer, primary_key=True)
    employee_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100))
    role = Column(String(50))  # 'employee' or 'admin'
    department = Column(String(100))
    consent_given = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    sessions = relationship("StressSession", back_populates="employee")
    stress_records = relationship("StressRecord", back_populates="employee")
    
    def __repr__(self):
        return f"<Employee {self.employee_id}>"


class StressSession(Base):
    """
    Monitoring session table
    Tracks individual analysis sessions
    """
    __tablename__ = 'sessions'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    employee_id = Column(Integer, ForeignKey('employees.id'), nullable=False)
    
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    duration_seconds = Column(Integer)
    
    # Session statistics
    avg_stress_score = Column(Float)
    max_stress_score = Column(Float)
    min_stress_score = Column(Float)
    total_records = Column(Integer, default=0)
    
    # High stress incidents
    high_stress_incidents = Column(Integer, default=0)
    alerts_triggered = Column(Integer, default=0)
    
    # Session metadata
    is_active = Column(Boolean, default=True)
    notes = Column(String(500))
    
    # Relationships
    employee = relationship("Employee", back_populates="sessions")
    stress_records = relationship("StressRecord", back_populates="session")
    
    def __repr__(self):
        return f"<Session {self.session_id}>"


class StressRecord(Base):
    """
    Individual stress measurement table
    Stores each analysis result with timestamp
    
    Privacy note: Raw video/audio NOT stored, only derived metrics
    """
    __tablename__ = 'stress_records'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=False, index=True)
    employee_id = Column(Integer, ForeignKey('employees.id'), nullable=False, index=True)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Facial analysis results
    facial_score = Column(Float)
    facial_emotion = Column(String(20))
    facial_confidence = Column(Float)
    face_detected = Column(Boolean)
    
    # Speech analysis results
    speech_score = Column(Float)
    speech_stress_level = Column(String(20))
    speech_confidence = Column(Float)
    
    # Fused results
    fused_score = Column(Float, nullable=False)
    smoothed_score = Column(Float, nullable=False)
    stress_category = Column(String(20), nullable=False)
    overall_confidence = Column(Float)
    
    # Metadata
    modality_used = Column(String(20))  # 'both', 'facial', 'speech'
    alert_triggered = Column(Boolean, default=False)
    
    # Relationships
    session = relationship("StressSession", back_populates="stress_records")
    employee = relationship("Employee", back_populates="stress_records")
    
    def __repr__(self):
        return f"<StressRecord {self.id} - {self.stress_category}>"


class Alert(Base):
    """
    Alert/notification table
    Tracks when stress thresholds are exceeded
    """
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey('employees.id'), nullable=False)
    session_id = Column(Integer, ForeignKey('sessions.id'))
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    alert_type = Column(String(50))  # 'high_stress', 'prolonged_stress', etc.
    severity = Column(String(20))  # 'low', 'medium', 'high'
    
    stress_score = Column(Float)
    duration_seconds = Column(Integer)
    
    message = Column(String(500))
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    
    def __repr__(self):
        return f"<Alert {self.alert_type} - {self.severity}>"


class AggregatedStats(Base):
    """
    Aggregated statistics table
    Pre-computed stats for admin dashboard (privacy-preserving)
    """
    __tablename__ = 'aggregated_stats'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.utcnow, index=True)
    department = Column(String(100), index=True)
    
    # Aggregated metrics (no individual identification)
    avg_stress_score = Column(Float)
    median_stress_score = Column(Float)
    max_stress_score = Column(Float)
    
    # Distribution counts
    low_stress_count = Column(Integer, default=0)
    medium_stress_count = Column(Integer, default=0)
    high_stress_count = Column(Integer, default=0)
    
    # Team size (for anonymization threshold check)
    employee_count = Column(Integer)
    total_sessions = Column(Integer)
    total_records = Column(Integer)
    
    # Alert statistics
    total_alerts = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<AggregatedStats {self.date} - {self.department}>"


# Database initialization
def init_database(database_url=None):
    """
    Initialize database and create all tables
    
    Args:
        database_url: Database connection string (uses config if None)
    """
    if database_url is None:
        database_url = DATABASE_URL
    
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    
    return engine


def get_session(engine=None):
    """
    Get database session
    
    Args:
        engine: SQLAlchemy engine (creates new if None)
        
    Returns:
        Database session
    """
    if engine is None:
        engine = create_engine(DATABASE_URL, echo=False)
    
    Session = sessionmaker(bind=engine)
    return Session()


# Helper functions for common operations

def create_employee(session, employee_id, name, role='employee', department=None):
    """Create new employee record"""
    employee = Employee(
        employee_id=employee_id,
        name=name,
        role=role,
        department=department
    )
    session.add(employee)
    session.commit()
    return employee


def create_session_record(session, employee_id, session_id):
    """Create new monitoring session"""
    stress_session = StressSession(
        session_id=session_id,
        employee_id=employee_id
    )
    session.add(stress_session)
    session.commit()
    return stress_session


def add_stress_record(session, session_id, employee_id, analysis_result):
    """
    Add stress analysis record
    
    Args:
        session: Database session
        session_id: Session ID
        employee_id: Employee ID
        analysis_result: Dictionary from multimodal fusion
    """
    record = StressRecord(
        session_id=session_id,
        employee_id=employee_id,
        facial_score=analysis_result.get('facial_score'),
        facial_emotion=analysis_result.get('facial_emotion'),
        facial_confidence=analysis_result.get('confidence'),
        face_detected=analysis_result.get('has_facial'),
        speech_score=analysis_result.get('speech_score'),
        speech_stress_level=analysis_result.get('speech_stress_level'),
        fused_score=analysis_result.get('fused_score'),
        smoothed_score=analysis_result.get('smoothed_score'),
        stress_category=analysis_result.get('stress_category'),
        overall_confidence=analysis_result.get('confidence'),
        timestamp=datetime.utcnow()
    )
    session.add(record)
    session.commit()
    return record


if __name__ == "__main__":
    # Initialize database
    engine = init_database()
    print("Database initialized successfully")
    print(f"Tables created: {', '.join(Base.metadata.tables.keys())}")
