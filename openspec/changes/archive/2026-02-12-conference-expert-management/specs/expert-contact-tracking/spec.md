## ADDED Requirements

### Requirement: Record contact history
The system SHALL allow users to record communication history with experts.

#### Scenario: Successfully add contact record
- **WHEN** user submits a contact record with expert ID, contact time, method, and content
- **THEN** system saves the contact record and associates it with the expert

#### Scenario: Add contact record for non-existent expert
- **WHEN** user attempts to add contact record for an expert that does not exist
- **THEN** system returns "Expert not found" error

### Requirement: View contact history
The system SHALL allow users to view all contact records for a specific expert.

#### Scenario: View contact history with records
- **WHEN** user requests contact history for an expert with existing records
- **THEN** system returns a list of all contact records sorted by time (newest first)

#### Scenario: View empty contact history
- **WHEN** user requests contact history for an expert with no records
- **THEN** system returns an empty list

### Requirement: Update contact status
The system SHALL allow users to update the contact status of an expert.

#### Scenario: Update contact status
- **WHEN** user updates expert's contact status (e.g., "待联系", "已联系", "确认参加", "暂不参与")
- **THEN** system saves the new status and records the update time
