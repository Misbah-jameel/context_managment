import asyncio
from pydantic import BaseModel
from agents import Agent, RunContextWrapper, Runner, function_tool
from connection import config
import rich

# 1. BANK ACCOUNT CONTEXT MODEL
class BankAccount(BaseModel):
    account_number: str
    account_name: str
    account_balance: float
    account_type: str

# 2. STUDENT PROFILE CONTEXT MODEL
class StudentProfile(BaseModel):
    student_id: str
    student_name: str
    student_semester: int
    total_course: int

# 3. LIBRARY BOOK CONTEXT MODEL
class LibraryBook(BaseModel):
    book_id: str
    book_title: str
    book_author: str
    is_available: bool

# Example Context Data
bank_account = BankAccount(
    account_number="ACC-789456",
    account_name="Fatima Khan",  
    account_balance=75500.50,
    account_type="savings"
)

student = StudentProfile(
    student_id="STU-456",
    student_name="Hassan Ahmed",
    student_semester=4,  
    total_course=5      
)

library_book = LibraryBook(
    book_id="BOOK-123",
    book_title="Python Programming",
    book_author="John Smith",  
    is_available=True
)

# ==========TOOLS==========
@function_tool
def get_bank_info(wrapper: RunContextWrapper[BankAccount]):
    b = wrapper.context
    return f"Account Holder: {b.account_name}, Account Number: {b.account_number}, Balance: {b.account_balance}, Type: {b.account_type}"

@function_tool
def get_student_info(wrapper: RunContextWrapper[StudentProfile]):
    s = wrapper.context
    return f"Student Name: {s.student_name}, ID: {s.student_id}, Semester: {s.student_semester}, Total Courses: {s.total_course}"

@function_tool
def get_book_info(wrapper: RunContextWrapper[LibraryBook]):
    bk = wrapper.context
    status = "Available" if bk.is_available else "Not Available"
    return f"Book: {bk.book_title} by {bk.book_author} (ID: {bk.book_id}) is {status}."

# ===== AGENTS =====
bank_agent = Agent(
    name="Bank Agent",
    instructions="You are a helpful bank assistant. Always call the tool to get bank account details.",
    tools=[get_bank_info]
)

student_agent = Agent(
    name="Student Agent",
    instructions="You are a helpful student assistant. Always call the tool to get student profile details.",
    tools=[get_student_info]
)

library_agent = Agent(
    name="Library Agent",
    instructions="You are a helpful library assistant. Always call the tool to get library book details.",
    tools=[get_book_info]
)

# ===== MAIN RUN FUNCTION =====
async def main():
    bank_result = await Runner.run(
        bank_agent,
        "Please tell me my account number and balance",
        run_config=config,
        context=bank_account
    )
    rich.print(bank_result.final_output)

    student_result = await Runner.run(
        student_agent,
        "What is my name and how many courses do I have?",
        run_config=config,
        context=student
    )
    rich.print(student_result.final_output)

    book_result = await Runner.run(
        library_agent,
        "Is the book available and who is the author?",
        run_config=config,
        context=library_book
    )
    rich.print(book_result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
