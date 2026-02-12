package com.example.expert.entity;

import jakarta.persistence.*;
import lombok.Data;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.LocalDate;
import java.time.LocalDateTime;

@Entity
@Table(name = "conference_records")
@Data
public class ConferenceRecord {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(name = "expert_id", nullable = false)
    private Long expertId;
    
    @Column(name = "conference_name", nullable = false, length = 255)
    private String conferenceName;
    
    @Column(name = "conference_date")
    private LocalDate conferenceDate;
    
    @Column(length = 100)
    private String role;
    
    @Column(length = 500)
    private String topic;
    
    @Column(columnDefinition = "TEXT")
    private String feedback;
    
    @Column(name = "is_completed")
    private Boolean isCompleted = false;
    
    @CreationTimestamp
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;
    
    @UpdateTimestamp
    @Column(name = "updated_at")
    private LocalDateTime updatedAt;
}
