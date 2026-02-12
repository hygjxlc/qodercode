package com.example.expert.controller;

import com.example.expert.dto.ContactRecordDTO;
import com.example.expert.service.ContactRecordService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/contacts")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class ContactRecordController {
    
    private final ContactRecordService contactRecordService;
    
    @PostMapping
    public ResponseEntity<?> addContactRecord(@Valid @RequestBody ContactRecordDTO dto) {
        try {
            ContactRecordDTO created = contactRecordService.addContactRecord(dto);
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("data", created);
            response.put("message", "Contact record added successfully");
            return ResponseEntity.status(HttpStatus.CREATED).body(response);
        } catch (RuntimeException e) {
            Map<String, Object> error = new HashMap<>();
            error.put("success", false);
            error.put("message", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
    
    @GetMapping("/expert/{expertId}")
    public ResponseEntity<?> getContactRecordsByExpertId(@PathVariable Long expertId) {
        try {
            List<ContactRecordDTO> records = contactRecordService.getContactRecordsByExpertId(expertId);
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
