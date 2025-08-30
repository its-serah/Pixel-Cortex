import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
from app.main import app
from app.core.database import get_db, Base
from app.core.security import get_password_hash
from app.models.models import User, UserRole

# Test database URL
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

@pytest.fixture(scope="session")
def engine():
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def db_session(engine):
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    yield session
    session.close()

@pytest.fixture(scope="function")
def client(db_session):
    def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()

@pytest.fixture
def test_user(db_session):
    user = User(
        username="testuser",
        email="test@example.com",
        full_name="Test User",
        role=UserRole.USER,
        hashed_password=get_password_hash("testpass"),
        is_active=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user

@pytest.fixture
def test_admin(db_session):
    admin = User(
        username="testadmin",
        email="admin@example.com",
        full_name="Test Admin",
        role=UserRole.ADMIN,
        hashed_password=get_password_hash("adminpass"),
        is_active=True
    )
    db_session.add(admin)
    db_session.commit()
    db_session.refresh(admin)
    return admin

@pytest.fixture
def test_agent(db_session):
    agent = User(
        username="testagent",
        email="agent@example.com",
        full_name="Test Agent",
        role=UserRole.AGENT,
        hashed_password=get_password_hash("agentpass"),
        is_active=True
    )
    db_session.add(agent)
    db_session.commit()
    db_session.refresh(agent)
    return agent

@pytest.fixture
def auth_headers(client, test_user):
    """Get auth headers for test user"""
    response = client.post("/api/auth/login", json={
        "username": "testuser",
        "password": "testpass"
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def admin_headers(client, test_admin):
    """Get auth headers for admin user"""
    response = client.post("/api/auth/login", json={
        "username": "testadmin",
        "password": "adminpass"
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
