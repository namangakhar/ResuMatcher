import os
from dotenv import load_dotenv
from google.adk.agents import Agent, SequentialAgent
from google.adk.tools.tool_context import ToolContext

load_dotenv()

model_name = os.getenv("MODEL", "gemini-2.5-flash")


# --- Tool: Save resume and JD to shared state ---
def save_resume_and_jd(tool_context: ToolContext, resume_text: str, job_description: str) -> dict:
    """Saves the user's resume and job description into shared state for downstream agents.

    Args:
        tool_context: The tool context for state management.
        resume_text: The candidate's full resume or LinkedIn profile text.
        job_description: The complete job description to match against.

    Returns:
        A confirmation that both inputs have been stored.
    """
    tool_context.state["RESUME"] = resume_text
    tool_context.state["JD"] = job_description
    return {"status": "success", "message": "Resume and JD saved. Starting analysis."}


# --- Agent 1: Analyzer - computes match score, strengths, gaps ---
analyzer_agent = Agent(
    name="analyzer",
    model=model_name,
    description="Analyzes resume against job description and produces a detailed match report.",
    instruction="""You are a resume-JD matching expert. Analyze the RESUME against the JD thoroughly.

Produce a detailed analysis in EXACTLY this format:

## 📊 Match Analysis

**Overall Match Score:** [X/100]

**Verdict:** [Strong Match / Moderate Match / Weak Match - brief explanation]

### ✅ Strengths (Skills & Experience that Align)
- [Specific strength referencing actual resume content and JD requirement]
- [Continue for all matches found]

### ⚠️ Gaps (What the JD Requires but Resume Lacks)
- [Specific gap — with concrete suggestion to address it]
- [Continue for all gaps found]

### 🔑 ATS Keyword Suggestions
Keywords from the JD to add to the profile:
- [Keyword 1]
- [Keyword 2]
- [Continue...]

Rules:
- Reference SPECIFIC skills and experiences from the resume, not generic statements.
- Be honest with the score. Don't inflate it.
- Every gap must have an actionable suggestion.

RESUME:
{RESUME}

JD:
{JD}
""",
    output_key="match_analysis",
)


# --- Agent 2: Cover Letter Writer ---
cover_letter_agent = Agent(
    name="cover_letter_writer",
    model=model_name,
    description="Generates a tailored cover letter based on the resume, JD, and match analysis.",
    instruction="""You are a professional cover letter writer. Using the candidate's resume, the job description,
and the match analysis, write a polished, ready-to-send cover letter.

The cover letter MUST:
- Be 3-4 paragraphs
- Open with genuine enthusiasm for the specific role (not generic)
- Highlight the candidate's strongest matching skills with concrete examples from their resume
- Acknowledge growth areas as opportunities, framing them positively
- Close with a confident call to action
- Sound human and natural — NOT like a template
- Use professional but warm tone

Format it as:

## 📝 Tailored Cover Letter

Dear Hiring Manager,

[Paragraph 1: Hook — enthusiasm for the role + strongest qualification]

[Paragraph 2: Body — 2-3 specific achievements/skills mapped to JD requirements]

[Paragraph 3: Address gaps positively + cultural fit / motivation]

[Paragraph 4: Confident close + call to action]

Sincerely,
[Candidate]

RESUME:
{RESUME}

JD:
{JD}

MATCH ANALYSIS:
{match_analysis}
""",
    output_key="cover_letter",
)


# --- Agent 3: Resume Optimizer ---
resume_optimizer_agent = Agent(
    name="resume_optimizer",
    model=model_name,
    description="Rewrites and optimizes the resume specifically tailored for the target JD.",
    instruction="""You are an expert resume writer and ATS optimization specialist.
Rewrite the candidate's resume to be PERFECTLY tailored for the target JD.

You MUST produce a complete, copy-paste ready resume in this format:

## 📄 Optimized Resume

### [CANDIDATE NAME]
[Contact info placeholder — Email | Phone | LinkedIn | Location]

---

#### PROFESSIONAL SUMMARY
[3-4 lines. Lead with years of experience + core expertise. Include top JD keywords naturally.
Frame everything toward the target role.]

#### TECHNICAL SKILLS
[Reorder to prioritize JD-relevant skills FIRST. Add suggested skills in brackets like [AWS - suggested].
Group by category: Languages, Frameworks, Cloud/DevOps, Databases, Tools]

#### PROFESSIONAL EXPERIENCE

**[Job Title]** | [Company] | [Dates]
- [Rewrite each bullet to emphasize JD-relevant achievements]
- [Use action verbs + metrics where possible]
- [Weave in JD keywords naturally]

#### EDUCATION
[Degree, University, Year]

#### CERTIFICATIONS (Suggested)
[List 2-3 relevant certifications the candidate should pursue based on gaps]

---

Rules:
- NEVER fabricate experience or skills the candidate doesn't have
- Mark suggested additions clearly with [suggested] tags
- Reorder and reframe existing content to maximize JD alignment
- Include ATS keywords from the JD naturally throughout
- Make it immediately copy-pasteable

RESUME:
{RESUME}

JD:
{JD}

MATCH ANALYSIS:
{match_analysis}
""",
    output_key="optimized_resume",
)


# --- Agent 4: Formatter - assembles final polished output ---
formatter_agent = Agent(
    name="formatter",
    model=model_name,
    description="Assembles all analysis into a clean, final output for the user.",
    instruction="""You are the final presenter. Take ALL the work done by previous agents and present it
as one clean, well-organized response. Do NOT re-analyze or change the content — just assemble it beautifully.

Present in this exact order:
1. The Match Analysis (from match_analysis)
2. The Tailored Cover Letter (from cover_letter)
3. The Optimized Resume (from optimized_resume)

Add a brief intro line like:
"Here's your complete ResuMatch analysis! Below you'll find your match report, a tailored cover letter, and an optimized resume — all ready to use. 🚀"

And a closing line like:
"💡 Need changes? Just ask — I can adjust the tone, focus on different skills, or rewrite any section!"

Keep ALL the original formatting, headers, and emojis intact. Don't summarize or truncate.

MATCH ANALYSIS:
{match_analysis}

COVER LETTER:
{cover_letter}

OPTIMIZED RESUME:
{optimized_resume}
""",
)


# --- Sequential Workflow: chains all 4 agents ---
analysis_workflow = SequentialAgent(
    name="analysis_workflow",
    description="Full resume-JD analysis pipeline: analyze → cover letter → optimize resume → format output.",
    sub_agents=[
        analyzer_agent,
        cover_letter_agent,
        resume_optimizer_agent,
        formatter_agent,
    ],
)


# --- Root Agent: Greeter that collects inputs then hands off ---
root_agent = Agent(
    name="resumatch_agent",
    model=model_name,
    description="ResuMatch — AI career advisor that analyzes resumes against job descriptions.",
    instruction="""You are **ResuMatch**, a friendly AI career advisor.

Your ONLY job is to collect two inputs from the user, then hand off to the analysis workflow.

## Conversation flow:

1. Greet the user warmly. Tell them you'll help match their resume to a job description.
   Ask them to provide their resume or LinkedIn profile text.

2. When they provide their resume, thank them and ask for the job description (JD).

3. Once you have BOTH the resume AND the JD, use the `save_resume_and_jd` tool to store them.
   After the tool confirms success, transfer to the `analysis_workflow` agent.

## Rules:
- Do NOT analyze anything yourself. Your only job is to collect resume + JD and hand off.
- Do NOT ask for the resume or JD more than once each. Once you have both, immediately use the tool and transfer.
- Be concise and friendly in your messages. Keep greetings short.
- If the user provides both resume and JD in a single message, that's fine — use the tool immediately.
""",
    tools=[save_resume_and_jd],
    sub_agents=[analysis_workflow],
)
