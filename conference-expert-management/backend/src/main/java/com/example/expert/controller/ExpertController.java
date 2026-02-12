package com.example.expert.controller;

import com.example.expert.dto.ExpertDTO;
import com.example.expert.service.ExpertService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/experts")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class ExpertController {
    
    private final ExpertService expertService;
    
    @PostMapping
    public ResponseEntity<?> createExpert(@Valid @RequestBody ExpertDTO expertDTO) {
        try {
            ExpertDTO created = expertService.createExpert(expertDTO);
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("data", created);
            response.put("message", "Expert created successfully");
            return ResponseEntity.status(HttpStatus.CREATED).body(response);
        } catch (RuntimeException e) {
            Map<String, Object> error = new HashMap<>();
            error.put("success", false);
            error.put("message", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
    
    @PutMapping("/{id}")
    public ResponseEntity<?> updateExpert(@PathVariable Long id, @Valid @RequestBody ExpertDTO expertDTO) {
        try {
            ExpertDTO updated = expertService.updateExpert(id, expertDTO);
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("data", updated);
            response.put("message", "Expert updated successfully");
            return ResponseEntity.ok(response);
        } catch (RuntimeException e) {
            Map<String, Object> error = new HashMap<>();
            error.put("success", false);
            error.put("message", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
    
    @DeleteMapping("/{id}")
    public ResponseEntity<?> deleteExpert(@PathVariable Long id) {
        try {
            expertService.deleteExpert(id);
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "Expert deleted successfully");
            return ResponseEntity.ok(response);
        } catch (RuntimeException e) {
            Map<String, Object> error = new HashMap<>();
            error.put("success", false);
            error.put("message", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
    
    @GetMapping("/{id}")
    public ResponseEntity<?> getExpertById(@PathVariable Long id) {
        try {
            ExpertDTO expert = expertService.getExpertById(id);
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("data", expert);
            return ResponseEntity.ok(response);
        } catch (RuntimeException e) {
            Map<String, Object> error = new HashMap<>();
            error.put("success", false);
            error.put("message", e.getMessage());
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(error);
        }
    }
    
    @GetMapping
    public ResponseEntity<?> getAllExperts(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {
        Pageable pageable = PageRequest.of(page, size);
        Page<ExpertDTO> experts = expertService.getAllExperts(pageable);
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("data", experts.getContent());
        response.put("totalElements", experts.getTotalElements());
        response.put("totalPages", experts.getTotalPages());
        response.put("currentPage", experts.getNumber());
        return ResponseEntity.ok(response);
    }
    
    @GetMapping("/search")
    public ResponseEntity<?> searchExperts(@RequestParam String keyword) {
        List<ExpertDTO> experts = expertService.searchExperts(keyword);
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("data", experts);
        return ResponseEntity.ok(response);
    }
    
    @GetMapping("/filter")
    public ResponseEntity<?> filterExperts(
            @RequestParam(required = false) String field,
            @RequestParam(required = false) String title,
            @RequestParam(required = false) String status) {
        List<ExpertDTO> experts = expertService.filterExperts(field, title, status);
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("data", experts);
        return ResponseEntity.ok(response);
    }
}
