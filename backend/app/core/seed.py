from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.core.security import get_password_hash
from app.models.models import User, UserRole, SLAConfig, TicketCategory, TicketPriority, Ticket
from datetime import datetime, timedelta
import os

def create_default_users(db: Session):
    """Create default admin and sample users"""
    
    # Admin user
    admin_email = os.getenv("ADMIN_EMAIL", "admin@pixelcortex.local")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
    
    existing_admin = db.query(User).filter(User.email == admin_email).first()
    if not existing_admin:
        admin_user = User(
            username="admin",
            email=admin_email,
            full_name="System Administrator",
            role=UserRole.ADMIN,
            hashed_password=get_password_hash(admin_password),
            is_active=True
        )
        db.add(admin_user)
        print(f"Created admin user: {admin_email}")
    
    # Sample agent
    existing_agent = db.query(User).filter(User.username == "agent1").first()
    if not existing_agent:
        agent_user = User(
            username="agent1",
            email="agent1@pixelcortex.local",
            full_name="IT Support Agent",
            role=UserRole.AGENT,
            hashed_password=get_password_hash("agent123"),
            is_active=True
        )
        db.add(agent_user)
        print("Created sample agent user")
    
    # Sample regular user
    existing_user = db.query(User).filter(User.username == "user1").first()
    if not existing_user:
        regular_user = User(
            username="user1",
            email="user1@pixelcortex.local",
            full_name="John Doe",
            role=UserRole.USER,
            hashed_password=get_password_hash("user123"),
            is_active=True
        )
        db.add(regular_user)
        print("Created sample regular user")
    
    db.commit()

def create_sla_config(db: Session):
    """Create default SLA configuration"""
    
    sla_configs = [
        # Critical priority SLAs
        (TicketCategory.SECURITY, TicketPriority.CRITICAL, 1, 4),
        (TicketCategory.NETWORK, TicketPriority.CRITICAL, 1, 8),
        (TicketCategory.HARDWARE, TicketPriority.CRITICAL, 2, 12),
        (TicketCategory.SOFTWARE, TicketPriority.CRITICAL, 2, 16),
        (TicketCategory.ACCESS, TicketPriority.CRITICAL, 1, 6),
        (TicketCategory.OTHER, TicketPriority.CRITICAL, 2, 24),
        
        # High priority SLAs
        (TicketCategory.SECURITY, TicketPriority.HIGH, 4, 24),
        (TicketCategory.NETWORK, TicketPriority.HIGH, 4, 24),
        (TicketCategory.HARDWARE, TicketPriority.HIGH, 8, 48),
        (TicketCategory.SOFTWARE, TicketPriority.HIGH, 8, 48),
        (TicketCategory.ACCESS, TicketPriority.HIGH, 4, 24),
        (TicketCategory.OTHER, TicketPriority.HIGH, 8, 72),
        
        # Medium priority SLAs
        (TicketCategory.SECURITY, TicketPriority.MEDIUM, 8, 72),
        (TicketCategory.NETWORK, TicketPriority.MEDIUM, 12, 72),
        (TicketCategory.HARDWARE, TicketPriority.MEDIUM, 24, 120),
        (TicketCategory.SOFTWARE, TicketPriority.MEDIUM, 24, 120),
        (TicketCategory.ACCESS, TicketPriority.MEDIUM, 12, 72),
        (TicketCategory.OTHER, TicketPriority.MEDIUM, 24, 168),
        
        # Low priority SLAs
        (TicketCategory.SECURITY, TicketPriority.LOW, 24, 168),
        (TicketCategory.NETWORK, TicketPriority.LOW, 48, 168),
        (TicketCategory.HARDWARE, TicketPriority.LOW, 72, 336),
        (TicketCategory.SOFTWARE, TicketPriority.LOW, 72, 336),
        (TicketCategory.ACCESS, TicketPriority.LOW, 48, 168),
        (TicketCategory.OTHER, TicketPriority.LOW, 72, 504),
    ]
    
    for category, priority, response_hours, resolution_hours in sla_configs:
        existing = db.query(SLAConfig).filter(
            SLAConfig.category == category,
            SLAConfig.priority == priority
        ).first()
        
        if not existing:
            sla = SLAConfig(
                category=category,
                priority=priority,
                response_time_hours=response_hours,
                resolution_time_hours=resolution_hours
            )
            db.add(sla)
    
    db.commit()
    print("Created SLA configuration")

def create_sample_tickets(db: Session):
    """Create sample tickets for demonstration"""
    
    # Get users
    admin = db.query(User).filter(User.username == "admin").first()
    agent = db.query(User).filter(User.username == "agent1").first()
    user = db.query(User).filter(User.username == "user1").first()
    
    if not all([admin, agent, user]):
        print("Users not found, skipping sample ticket creation")
        return
    
    sample_tickets = [
        {
            "title": "Computer won't turn on",
            "description": "My desktop computer stopped working this morning. The power button doesn't respond and there are no lights on the case. I have an important presentation tomorrow.",
            "category": TicketCategory.HARDWARE,
            "priority": TicketPriority.HIGH,
            "requester_id": user.id
        },
        {
            "title": "Can't access email",
            "description": "I forgot my password and can't log into my email account. I tried the password reset but didn't receive the email.",
            "category": TicketCategory.ACCESS,
            "priority": TicketPriority.MEDIUM,
            "requester_id": user.id
        },
        {
            "title": "Suspicious email received",
            "description": "I received an email asking for my login credentials. It looks like phishing. The sender appears to be from IT but the email address looks suspicious.",
            "category": TicketCategory.SECURITY,
            "priority": TicketPriority.HIGH,
            "requester_id": user.id,
            "assigned_agent_id": agent.id
        },
        {
            "title": "Software installation request",
            "description": "I need Adobe Photoshop installed on my workstation for design work. Please let me know the process.",
            "category": TicketCategory.SOFTWARE,
            "priority": TicketPriority.LOW,
            "requester_id": user.id
        },
        {
            "title": "Internet connection issues",
            "description": "The internet connection in our office is very slow today. Multiple users are affected. Speed test shows very low bandwidth.",
            "category": TicketCategory.NETWORK,
            "priority": TicketPriority.HIGH,
            "requester_id": user.id,
            "assigned_agent_id": agent.id
        }
    ]
    
    for ticket_data in sample_tickets:
        existing = db.query(Ticket).filter(Ticket.title == ticket_data["title"]).first()
        if not existing:
            # Calculate due date based on SLA
            sla = db.query(SLAConfig).filter(
                SLAConfig.category == ticket_data["category"],
                SLAConfig.priority == ticket_data["priority"]
            ).first()
            
            due_date = None
            if sla:
                due_date = datetime.utcnow() + timedelta(hours=sla.resolution_time_hours)
            
            ticket = Ticket(
                title=ticket_data["title"],
                description=ticket_data["description"],
                category=ticket_data["category"],
                priority=ticket_data["priority"],
                requester_id=ticket_data["requester_id"],
                assigned_agent_id=ticket_data.get("assigned_agent_id"),
                due_date=due_date
            )
            db.add(ticket)
    
    db.commit()
    print("Created sample tickets")

def seed_database():
    """Main seeding function"""
    db = SessionLocal()
    try:
        print("Seeding database...")
        create_default_users(db)
        create_sla_config(db)
        create_sample_tickets(db)
        print("Database seeding completed successfully")
    except Exception as e:
        print(f"Error seeding database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
