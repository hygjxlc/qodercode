package com.example.expert.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

import java.time.LocalDateTime;

@Data
public class ExpertDTO {
    
    private Long id;
    
    @NotBlank(message = "姓名不能为空")
    private String name;
    
    @NotBlank(message = "邮箱不能为空")
    @Email(message = "邮箱格式不正确")
    private String email;
    
    private String phone;
    
    private String professionalField;
    
    private String title;
    
    private String organization;
    
    private String address;
    
    private String bio;
    
    private String contactStatus;
    
    private LocalDateTime createdAt;
    
    private LocalDateTime updatedAt;
}
