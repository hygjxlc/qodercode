## ADDED Requirements

### Requirement: Record conference participation
The system SHALL allow users to record expert's participation in conferences.

#### Scenario: Add conference participation record
- **WHEN** user adds a conference record with conference name, date, role, and topic
- **THEN** system saves the record and associates it with the expert

#### Scenario: Add duplicate conference record
- **WHEN** user attempts to add a conference record that already exists for the expert
- **THEN** system updates the existing record with new information

### Requirement: Record expert feedback
The system SHALL allow users to record feedback from experts after conferences.

#### Scenario: Add feedback to conference record
- **WHEN** user submits feedback for a conference participation record
- **THEN** system saves the feedback and marks the record as completed

### Requirement: View conference history
The system SHALL allow users to view all conferences an expert has participated in.

#### Scenario: View conference history
- **WHEN** user requests conference history for an expert
- **THEN** system returns a list of all conference participations with details
