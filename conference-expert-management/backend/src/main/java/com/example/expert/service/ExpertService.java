package com.example.expert.service;

import com.example.expert.dto.ExpertDTO;
import com.example.expert.entity.Expert;
import com.example.expert.repository.ExpertRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.BeanUtils;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class ExpertService {
    
    private final ExpertRepository expertRepository;
    
    @Transactional
    public ExpertDTO createExpert(ExpertDTO expertDTO) {
        if (expertRepository.existsByEmailAndIsDeletedFalse(expertDTO.getEmail())) {
            throw new RuntimeException("Email already registered");
        }
        
        Expert expert = new Expert();
        BeanUtils.copyProperties(expertDTO, expert);
        expert.setIsDeleted(false);
        
        Expert saved = expertRepository.save(expert);
        return convertToDTO(saved);
    }
    
    @Transactional
    public ExpertDTO updateExpert(Long id, ExpertDTO expertDTO) {
        Expert expert = expertRepository.findByIdAndIsDeletedFalse(id)
                .orElseThrow(() -> new RuntimeException("Expert not found"));
        
        // Check email uniqueness if changed
        if (!expert.getEmail().equals(expertDTO.getEmail()) && 
            expertRepository.existsByEmailAndIsDeletedFalse(expertDTO.getEmail())) {
            throw new RuntimeException("Email already registered");
        }
        
        BeanUtils.copyProperties(expertDTO, expert, "id", "createdAt", "isDeleted");
        Expert updated = expertRepository.save(expert);
        return convertToDTO(updated);
    }
    
    @Transactional
    public void deleteExpert(Long id) {
        Expert expert = expertRepository.findByIdAndIsDeletedFalse(id)
                .orElseThrow(() -> new RuntimeException("Expert not found"));
        expert.setIsDeleted(true);
        expertRepository.save(expert);
    }
    
    @Transactional(readOnly = true)
    public ExpertDTO getExpertById(Long id) {
        Expert expert = expertRepository.findByIdAndIsDeletedFalse(id)
                .orElseThrow(() -> new RuntimeException("Expert not found"));
        return convertToDTO(expert);
    }
    
    @Transactional(readOnly = true)
    public Page<ExpertDTO> getAllExperts(Pageable pageable) {
        return expertRepository.findByIsDeletedFalse(pageable)
                .map(this::convertToDTO);
    }
    
    @Transactional(readOnly = true)
    public List<ExpertDTO> searchExperts(String keyword) {
        return expertRepository.searchByKeyword(keyword)
                .stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }
    
    @Transactional(readOnly = true)
    public List<ExpertDTO> filterExperts(String field, String title, String status) {
        return expertRepository.filterExperts(field, title, status)
                .stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }
    
    private ExpertDTO convertToDTO(Expert expert) {
        ExpertDTO dto = new ExpertDTO();
        BeanUtils.copyProperties(expert, dto);
        return dto;
    }
}
