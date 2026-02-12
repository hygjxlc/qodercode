package com.example.expert.entity;

import jakarta.persistence.*;
import lombok.Data;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.LocalDateTime;

@Entity
@Table(name = "experts")
@Data
public class Expert {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(nullable = false, length = 100)
    private String name;
    
    @Column(nullable = false, unique = true, length = 255)
    private String email;
    
    @Column(length = 20)
    private String phone;
    
    @Column(name = "professional_field", length = 255)
    private String professionalField;
    
    @Column(length = 100)
    private String title;
    
    @Column(length = 255)
    private String organization;
    
    @Column(length = 500)
    private String address;
    
    @Column(columnDefinition = "TEXT")
    private String bio;
    
    @Column(name = "contact_status", length = 50)
    private String contactStatus = "待联系";
    
    @Column(name = "is_deleted")
    private Boolean isDeleted = false;
    
    @CreationTimestamp
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;
    
    @UpdateTimestamp
    @Column(name = "updated_at")
    private LocalDateTime updatedAt;
}
