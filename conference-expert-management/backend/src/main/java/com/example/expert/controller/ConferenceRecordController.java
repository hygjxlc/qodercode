package com.example.expert.controller;

import com.example.expert.dto.ConferenceRecordDTO;
import com.example.expert.service.ConferenceRecordService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/conferences")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class ConferenceRecordController {
    
    private final ConferenceRecordService conferenceRecordService;
    
    @PostMapping
    public ResponseEntity<?> addConferenceRecord(@Valid @RequestBody ConferenceRecordDTO dto) {
        try {
            ConferenceRecordDTO created = conferenceRecordService.addConferenceRecord(dto);
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("data", created);
            response.put("message", "Conference record added successfully");
            return ResponseEntity.status(HttpStatus.CREATED).body(response);
        } catch (RuntimeException e) {
            Map<String, Object> error = new HashMap<>();
            error.put("success", false);
            error.put("message", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
    
    @PostMapping("/{id}/feedback")
    public ResponseEntity<?> addFeedback(@PathVariable Long id, @RequestBody Map<String, String> body) {
        try {
            String feedback = body.get("feedback");
            ConferenceRecordDTO updated = conferenceRecordService.addFeedback(id, feedback);
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("data", updated);
            response.put("message", "Feedback added successfully");
            return ResponseEntity.ok(response);
        } catch (RuntimeException e) {
            Map<String, Object> error = new HashMap<>();
            error.put("success", false);
            error.put("message", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
    
    @GetMapping("/expert/{expertId}")
    public ResponseEntity<?> getConferenceRecordsByExpertId(@PathVariable Long expertId) {
        try {
            List<ConferenceRecordDTO> records = conferenceRecordService.getConferenceRecordsByExpertId(expertId);
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("data", records);
            return ResponseEntity.ok(response);
        } catch (RuntimeException e) {
            Map<String, Object> error = new HashMap<>();
            error.put("success", false);
            error.put("message", e.getMessage());
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(error);
        }
    }
}
