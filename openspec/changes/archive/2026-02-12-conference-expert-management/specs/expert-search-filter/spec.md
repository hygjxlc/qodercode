## ADDED Requirements

### Requirement: Search experts by keyword
The system SHALL allow users to search experts using keywords.

#### Scenario: Search by name
- **WHEN** user enters a keyword matching expert's name
- **THEN** system returns all experts whose name contains the keyword

#### Scenario: Search by professional field
- **WHEN** user enters a keyword matching expert's professional field
- **THEN** system returns all experts in that field

#### Scenario: Search with no results
- **WHEN** user searches with a keyword that matches no experts
- **THEN** system returns an empty list

### Requirement: Filter experts by criteria
The system SHALL allow users to filter experts using multiple criteria.

#### Scenario: Filter by single criterion
- **WHEN** user applies a filter (e.g., professional field = "人工智能")
- **THEN** system returns experts matching that criterion

#### Scenario: Filter by multiple criteria
- **WHEN** user applies multiple filters (e.g., field = "人工智能" AND title = "教授")
- **THEN** system returns experts matching all criteria

#### Scenario: Combine search and filter
- **WHEN** user enters a keyword and applies filters simultaneously
- **THEN** system returns experts matching both the keyword and filter criteria

### Requirement: Sort search results
The system SHALL allow users to sort search results.

#### Scenario: Sort by name
- **WHEN** user chooses to sort results by name
- **THEN** system returns results sorted alphabetically by name

#### Scenario: Sort by last contact time
- **WHEN** user chooses to sort by last contact time
- **THEN** system returns results sorted by most recent contact first
