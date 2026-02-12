package com.example.expert.entity;

import jakarta.persistence.*;
import lombok.Data;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;

@Entity
@Table(name = "contact_records")
@Data
public class ContactRecord {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(name = "expert_id", nullable = false)
    private Long expertId;
    
    @Column(name = "contact_time", nullable = false)
    private LocalDateTime contactTime;
    
    @Column(name = "contact_method", length = 50)
    private String contactMethod;
    
    @Column(columnDefinition = "TEXT")
    private String content;
    
    @Column(length = 100)
    private String operator;
    
    @CreationTimestamp
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;
}
