package com.example.expert.controller;

import com.example.expert.repository.ExpertRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/statistics")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class StatisticsController {
    
    private final ExpertRepository expertRepository;
    
    @GetMapping("/overview")
    public ResponseEntity<?> getOverview() {
        long totalExperts = expertRepository.countByIsDeletedFalse();
        
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("data", Map.of(
                "totalExperts", totalExperts,
                "contactStatusDistribution", getContactStatusDistribution(),
                "fieldDistribution", getFieldDistribution()
        ));
        return ResponseEntity.ok(response);
    }
    
    private Map<String, Long> getContactStatusDistribution() {
        return expertRepository.findByIsDeletedFalse(org.springframework.data.domain.Pageable.unpaged())
                .getContent()
                .stream()
                .collect(Collectors.groupingBy(
                        expert -> expert.getContactStatus(),
                        Collectors.counting()
                ));
    }
    
    private Map<String, Long> getFieldDistribution() {
        return expertRepository.findByIsDeletedFalse(org.springframework.data.domain.Pageable.unpaged())
                .getContent()
                .stream()
                .filter(expert -> expert.getProfessionalField() != null)
                .collect(Collectors.groupingBy(
                        expert -> expert.getProfessionalField(),
                        Collectors.counting()
                ));
    }
}
