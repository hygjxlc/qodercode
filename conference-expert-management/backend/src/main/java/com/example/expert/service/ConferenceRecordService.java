package com.example.expert.service;

import com.example.expert.dto.ConferenceRecordDTO;
import com.example.expert.entity.ConferenceRecord;
import com.example.expert.repository.ConferenceRecordRepository;
import com.example.expert.repository.ExpertRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class ConferenceRecordService {
    
    private final ConferenceRecordRepository conferenceRecordRepository;
    private final ExpertRepository expertRepository;
    
    @Transactional
    public ConferenceRecordDTO addConferenceRecord(ConferenceRecordDTO dto) {
        // Verify expert exists
        if (!expertRepository.existsById(dto.getExpertId())) {
            throw new RuntimeException("Expert not found");
        }
        
        // Check for duplicate
        if (dto.getConferenceDate() != null) {
            var existing = conferenceRecordRepository
                    .findByExpertIdAndConferenceNameAndConferenceDate(
                            dto.getExpertId(), dto.getConferenceName(), dto.getConferenceDate());
            
            if (existing.isPresent()) {
                // Update existing record
                ConferenceRecord record = existing.get();
                BeanUtils.copyProperties(dto, record, "id", "createdAt");
                ConferenceRecord updated = conferenceRecordRepository.save(record);
                return convertToDTO(updated);
            }
        }
        
        // Create new record
        ConferenceRecord record = new ConferenceRecord();
        BeanUtils.copyProperties(dto, record);
        record.setIsCompleted(false);
        
        ConferenceRecord saved = conferenceRecordRepository.save(record);
        return convertToDTO(saved);
    }
    
    @Transactional
    public ConferenceRecordDTO addFeedback(Long id, String feedback) {
        ConferenceRecord record = conferenceRecordRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Conference record not found"));
        
        record.setFeedback(feedback);
        record.setIsCompleted(true);
        
        ConferenceRecord updated = conferenceRecordRepository.save(record);
        return convertToDTO(updated);
    }
    
    @Transactional(readOnly = true)
    public List<ConferenceRecordDTO> getConferenceRecordsByExpertId(Long expertId) {
        if (!expertRepository.existsById(expertId)) {
            throw new RuntimeException("Expert not found");
        }
        
        return conferenceRecordRepository.findByExpertIdOrderByConferenceDateDesc(expertId)
                .stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }
    
    private ConferenceRecordDTO convertToDTO(ConferenceRecord record) {
        ConferenceRecordDTO dto = new ConferenceRecordDTO();
        BeanUtils.copyProperties(record, dto);
        return dto;
    }
}
