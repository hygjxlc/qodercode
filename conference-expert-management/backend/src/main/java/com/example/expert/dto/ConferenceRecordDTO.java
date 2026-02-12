package com.example.expert.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
public class ConferenceRecordDTO {
    
    private Long id;
    
    @NotNull(message = "专家ID不能为空")
    private Long expertId;
    
    @NotBlank(message = "会议名称不能为空")
    private String conferenceName;
    
    private LocalDate conferenceDate;
    
    private String role;
    
    private String topic;
    
    private String feedback;
    
    private Boolean isCompleted;
    
    private LocalDateTime createdAt;
    
    private LocalDateTime updatedAt;
}
