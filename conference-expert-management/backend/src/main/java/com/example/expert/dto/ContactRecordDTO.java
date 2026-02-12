package com.example.expert.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.time.LocalDateTime;

@Data
public class ContactRecordDTO {
    
    private Long id;
    
    @NotNull(message = "专家ID不能为空")
    private Long expertId;
    
    @NotNull(message = "联系时间不能为空")
    private LocalDateTime contactTime;
    
    private String contactMethod;
    
    private String content;
    
    private String operator;
    
    private LocalDateTime createdAt;
}
