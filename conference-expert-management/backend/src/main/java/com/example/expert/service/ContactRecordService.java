package com.example.expert.service;

import com.example.expert.dto.ContactRecordDTO;
import com.example.expert.entity.ContactRecord;
import com.example.expert.entity.Expert;
import com.example.expert.repository.ContactRecordRepository;
import com.example.expert.repository.ExpertRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class ContactRecordService {
    
    private final ContactRecordRepository contactRecordRepository;
    private final ExpertRepository expertRepository;
    
    @Transactional
    public ContactRecordDTO addContactRecord(ContactRecordDTO dto) {
        // Verify expert exists
        Expert expert = expertRepository.findByIdAndIsDeletedFalse(dto.getExpertId())
                .orElseThrow(() -> new RuntimeException("Expert not found"));
        
        ContactRecord record = new ContactRecord();
        BeanUtils.copyProperties(dto, record);
        
        ContactRecord saved = contactRecordRepository.save(record);
        
        // Update expert contact status if provided
        if (dto.getContent() != null && dto.getContent().contains("确认参加")) {
            expert.setContactStatus("确认参加");
            expertRepository.save(expert);
        } else if (dto.getContent() != null && dto.getContent().contains("暂不参与")) {
            expert.setContactStatus("暂不参与");
            expertRepository.save(expert);
        } else {
            expert.setContactStatus("已联系");
            expertRepository.save(expert);
        }
        
        return convertToDTO(saved);
    }
    
    @Transactional(readOnly = true)
    public List<ContactRecordDTO> getContactRecordsByExpertId(Long expertId) {
        // Verify expert exists
        if (!expertRepository.existsById(expertId)) {
            throw new RuntimeException("Expert not found");
        }
        
        return contactRecordRepository.findByExpertIdOrderByContactTimeDesc(expertId)
                .stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }
    
    private ContactRecordDTO convertToDTO(ContactRecord record) {
        ContactRecordDTO dto = new ContactRecordDTO();
        BeanUtils.copyProperties(record, dto);
        return dto;
    }
}
